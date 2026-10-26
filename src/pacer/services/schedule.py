"""The weekly schedule and attendance: who may do what, and the attendance rates.

    ScheduleService     the template (manager), publishing a week from it, changing one week
    AttendanceService   marking a session, and the rates per trainee and per group

Rules (decision 2026-10-26):
* The schedule is not sensitive: every staff member sees it. Only the manager changes it.
* Who is expected at a session: the ACTIVE trainees of its cohort, or of its group. (A trainee on
  leave or who left is not expected; the rule uses the trainee's status today, a simplification.)
* Who marks: a counselor marks the trainees of the groups they lead; the manager marks everyone.
  A trainee the actor may not see is 404, as everywhere (visibility.py).
* A session is marked on its day or after it, never before; a cancelled session is never marked.
* Rate = present / (present + absent). Excused is not counted against the trainee; a session that
  was not marked yet is "unmarked", and is not counted as either. None when nothing counts yet.
"""

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, time, timedelta

from pacer.domain import (
    AttendanceRecord,
    AttendanceStatus,
    ScheduledSession,
    ScheduleSlot,
    StaffMember,
    Trainee,
    TraineeStatus,
    Weekday,
)
from pacer.repositories.interfaces import (
    AttendanceRepo,
    CohortRepo,
    GroupRepo,
    ScheduledSessionRepo,
    ScheduleSlotRepo,
    TraineeRepo,
)

from .errors import RuleViolationError, found
from .permissions import can_view_trainee, is_manager, require
from .visibility import hidden, visible_trainee

log = logging.getLogger(__name__)


def _check_group(groups: GroupRepo, cohort_id: int, group_id: int | None) -> None:
    if group_id is None:
        return
    group = found(groups.get(group_id), "Group", group_id)
    if group.cohort_id != cohort_id:
        raise RuleViolationError(f"Group {group_id} belongs to cohort {group.cohort_id}, not {cohort_id}")


def _sunday(day: date) -> date:
    if Weekday(day.weekday()) is not Weekday.SUNDAY:
        raise RuleViolationError(f"A week starts on Sunday; {day} is a {Weekday(day.weekday()).name.lower()}")
    return day


class ScheduleService:
    def __init__(
        self, slots: ScheduleSlotRepo, sessions: ScheduledSessionRepo, cohorts: CohortRepo, groups: GroupRepo
    ) -> None:
        self._slots = slots
        self._sessions = sessions
        self._cohorts = cohorts
        self._groups = groups

    # --- the template -----------------------------------------------------------------------

    def add_slot(self, actor: StaffMember, slot: ScheduleSlot) -> ScheduleSlot:
        require(is_manager(actor), "change the schedule")
        found(self._cohorts.get(slot.cohort_id), "Cohort", slot.cohort_id)
        _check_group(self._groups, slot.cohort_id, slot.group_id)
        added = self._slots.add(slot)
        log.info("Schedule slot %s added to cohort %s by staff %s", added.id, slot.cohort_id, actor.id)
        return added

    def slots(self, cohort_id: int) -> list[ScheduleSlot]:
        """The template, active slots only, in the academy's week order (Sunday first)."""
        active = [s for s in self._slots.list_by_cohort(cohort_id) if s.active]
        return sorted(active, key=lambda s: (s.weekday.offset, s.start, s.id))

    def deactivate_slot(self, actor: StaffMember, slot_id: int) -> ScheduleSlot:
        """No longer published. The weeks already made from it stay as they were."""
        require(is_manager(actor), "change the schedule")
        slot = found(self._slots.get(slot_id), "Schedule slot", slot_id)
        if not slot.active:
            raise RuleViolationError(f"Schedule slot {slot_id} is already inactive")
        return self._slots.update(slot.model_copy(update={"active": False}))

    # --- weeks ------------------------------------------------------------------------------

    def publish_week(self, actor: StaffMember, cohort_id: int, sunday: date) -> list[ScheduledSession]:
        """This week's sessions from the template. Safe to run again: a slot that already has its
        session this week is skipped, so a slot added to the template later can be published too."""
        require(is_manager(actor), "change the schedule")
        cohort = found(self._cohorts.get(cohort_id), "Cohort", cohort_id)
        _sunday(sunday)
        saturday = sunday + timedelta(days=6)
        if saturday < cohort.start_date or sunday > cohort.end_date:
            dates = f"{cohort.start_date} to {cohort.end_date}"
            raise RuleViolationError(f"The week of {sunday} is outside the cohort ({dates})")
        slots = self.slots(cohort_id)
        made = self._sessions.slot_dates((s.key for s in slots), sunday, saturday)
        created = []
        for slot in slots:
            day = slot.on_week(sunday)
            if (slot.key, day) in made or not cohort.start_date <= day <= cohort.end_date:
                continue
            created.append(self._sessions.add(ScheduledSession.from_slot(slot, sunday)))
        log.info("Week of %s published for cohort %s: %s new sessions", sunday, cohort_id, len(created))
        return created

    def week(self, cohort_id: int, sunday: date) -> list[ScheduledSession]:
        """The week's sessions, cancelled ones too (marked as such), by day and time."""
        found(self._cohorts.get(cohort_id), "Cohort", cohort_id)
        return self._sessions.list_between(cohort_id, _sunday(sunday), sunday + timedelta(days=6))

    # --- one week's changes -----------------------------------------------------------------

    def add_session(self, actor: StaffMember, session: ScheduledSession) -> ScheduledSession:
        """A one-off: a trip, a guest lecture. It is not part of the template."""
        require(is_manager(actor), "change the schedule")
        found(self._cohorts.get(session.cohort_id), "Cohort", session.cohort_id)
        _check_group(self._groups, session.cohort_id, session.group_id)
        if session.slot_id is not None:
            raise RuleViolationError("A one-off session has no template slot")
        return self._sessions.add(session)

    def change_session(
        self,
        actor: StaffMember,
        session_id: int,
        *,
        title: str | None = None,
        start: time | None = None,
        end: time | None = None,
        on: date | None = None,
    ) -> ScheduledSession:
        require(is_manager(actor), "change the schedule")
        session = found(self._sessions.get(session_id), "Session", session_id)
        changes = {k: v for k, v in {"title": title, "start": start, "end": end, "on": on}.items() if v is not None}
        return self._sessions.update(session.changed(**changes))

    def cancel_session(self, actor: StaffMember, session_id: int) -> ScheduledSession:
        """Cancelled sessions take no attendance and are left out of every rate, also what was marked."""
        require(is_manager(actor), "change the schedule")
        session = found(self._sessions.get(session_id), "Session", session_id)
        return self._sessions.update(session.cancel())


# --- attendance ------------------------------------------------------------------------------


@dataclass(frozen=True)
class SheetRow:
    trainee: Trainee
    status: AttendanceStatus | None  # None: not marked yet


@dataclass(frozen=True)
class AttendanceSummary:
    trainee: Trainee
    present: int
    absent: int
    excused: int
    unmarked: int  # sessions that already happened and nobody marked
    missed_by_title: dict[str, int] = field(default_factory=dict)  # absent (not excused), by session title

    @property
    def counted(self) -> int:
        return self.present + self.absent

    @property
    def rate(self) -> float | None:
        """present / (present + absent): excused and unmarked do not count. None if nothing counts yet."""
        return None if self.counted == 0 else self.present / self.counted


@dataclass(frozen=True)
class Mark:
    trainee_id: int
    status: AttendanceStatus


class AttendanceService:
    def __init__(
        self,
        sessions: ScheduledSessionRepo,
        attendance: AttendanceRepo,
        trainees: TraineeRepo,
        groups: GroupRepo,
    ) -> None:
        self._sessions = sessions
        self._attendance = attendance
        self._trainees = trainees
        self._groups = groups

    def _expected(self, session: ScheduledSession) -> list[Trainee]:
        """The active trainees of the session's group, or of its cohort."""
        if session.group_id is not None:
            candidates = self._trainees.list_by_group(session.group_id)
        else:
            candidates = [t for t in self._trainees.list_all() if t.cohort_id == session.cohort_id]
        return sorted((t for t in candidates if t.status is TraineeStatus.ACTIVE), key=lambda t: t.key)

    def _visible(self, actor: StaffMember, trainees: Sequence[Trainee]) -> list[Trainee]:
        allowed: dict[int, bool] = {}

        def sees(trainee: Trainee) -> bool:
            if trainee.group_id not in allowed:
                group = found(self._groups.get(trainee.group_id), "Group", trainee.group_id)
                allowed[trainee.group_id] = can_view_trainee(actor, group)
            return allowed[trainee.group_id]

        return [t for t in trainees if sees(t)]

    def sheet(self, actor: StaffMember, session_id: int) -> tuple[ScheduledSession, list[SheetRow]]:
        """The session, and the trainees this actor marks there, with what was marked so far."""
        session = found(self._sessions.get(session_id), "Session", session_id)
        marked = {r.trainee_id: r.status for r in self._attendance.list_by_session(session_id)}
        rows = [SheetRow(t, marked.get(t.key)) for t in self._visible(actor, self._expected(session))]
        return session, rows

    def mark(
        self, actor: StaffMember, session_id: int, marks: Sequence[Mark], *, today: date
    ) -> list[AttendanceRecord]:
        """All marks or none (the caller's unit of work). Marking a trainee again replaces the mark."""
        session = found(self._sessions.get(session_id), "Session", session_id)
        if session.cancelled:
            raise RuleViolationError(f"Session {session_id} was cancelled: there is no attendance to mark")
        if session.on > today:
            raise RuleViolationError(f"Session {session_id} is on {session.on}: attendance is marked from that day on")
        ids = [m.trainee_id for m in marks]
        if duplicates := sorted({i for i in ids if ids.count(i) > 1}):
            raise RuleViolationError(f"Trainee marked twice in one request: {duplicates}")
        expected = {t.key for t in self._expected(session)}
        saved = []
        for mark in marks:
            visible_trainee(actor, self._trainees, self._groups, mark.trainee_id)  # 404 for a trainee not theirs
            if mark.trainee_id not in expected:
                raise RuleViolationError(f"Trainee {mark.trainee_id} is not expected at session {session_id}")
            saved.append(
                self._attendance.save(
                    AttendanceRecord(
                        session_id=session.key, trainee_id=mark.trainee_id, status=mark.status, recorded_by_id=actor.key
                    )
                )
            )
        log.info("Attendance: session %s, %s marks by staff %s", session_id, len(saved), actor.id)
        return saved

    # --- rates ------------------------------------------------------------------------------

    def _summaries(self, trainees: Sequence[Trainee], first: date, last: date, today: date) -> list[AttendanceSummary]:
        if not trainees:
            return []
        last_counted = min(last, today)  # a future session is not "unmarked"
        cohorts = {t.cohort_id for t in trainees}
        sessions = [
            s
            for cohort_id in cohorts
            for s in self._sessions.list_between(cohort_id, first, last_counted)
            if not s.cancelled
        ]
        by_id = {s.key: s for s in sessions}
        records: dict[tuple[int, int], AttendanceStatus] = {
            (r.trainee_id, r.session_id): r.status
            for r in self._attendance.list_for_trainees((t.key for t in trainees), first, last_counted)
            if r.session_id in by_id  # a cancelled session's marks do not count
        }
        result = []
        for trainee in trainees:
            mine = [s for s in sessions if s.cohort_id == trainee.cohort_id and s.group_id in (None, trainee.group_id)]
            counts: Counter[AttendanceStatus | None] = Counter()
            missed: Counter[str] = Counter()
            for session in mine:
                status = records.get((trainee.key, session.key))
                counts[status] += 1
                if status is AttendanceStatus.ABSENT:
                    missed[session.title] += 1
            result.append(
                AttendanceSummary(
                    trainee=trainee,
                    present=counts[AttendanceStatus.PRESENT],
                    absent=counts[AttendanceStatus.ABSENT],
                    excused=counts[AttendanceStatus.EXCUSED],
                    unmarked=counts[None],
                    missed_by_title=dict(missed.most_common()),
                )
            )
        return result

    def trainee_summary(
        self, actor: StaffMember, trainee_id: int, first: date, last: date, *, today: date
    ) -> AttendanceSummary:
        if last < first:
            raise RuleViolationError("The range ends before it starts")
        trainee = visible_trainee(actor, self._trainees, self._groups, trainee_id)
        return self._summaries([trainee], first, last, today)[0]

    def group_summary(
        self,
        actor: StaffMember,
        cohort_id: int,
        first: date,
        last: date,
        *,
        today: date,
        group_id: int | None = None,
    ) -> list[AttendanceSummary]:
        """Every active trainee this actor may see (of one group, if given), the lowest rate first:
        who needs attention is at the top. Trainees with nothing counted yet come last."""
        if last < first:
            raise RuleViolationError("The range ends before it starts")
        if group_id is not None:
            group = found(self._groups.get(group_id), "Group", group_id)
            if not can_view_trainee(actor, group):
                raise hidden("Group", group_id)
        trainees = [
            t
            for t in self._trainees.list_all()
            if t.cohort_id == cohort_id and t.is_active and (group_id is None or t.group_id == group_id)
        ]
        summaries = self._summaries(self._visible(actor, trainees), first, last, today)
        return sorted(summaries, key=lambda s: (s.rate is None, s.rate if s.rate is not None else 0, s.trainee.key))
