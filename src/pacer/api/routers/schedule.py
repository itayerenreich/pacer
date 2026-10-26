"""Endpoints under /schedule and /attendance: the weekly schedule, and who attended it.

Every staff member sees the schedule; only the manager changes it. Counselors mark the trainees of
their groups, the manager everyone. "Today" is the academy's day (Asia/Jerusalem), from get_today.
"""

from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Query, status

from pacer.domain import ScheduledSession, ScheduleSlot, week_start
from pacer.services import Mark

from ..dependencies import Actor, AttendanceServiceDep, ScheduleServiceDep, TodayDep
from ..schemas import (
    AttendanceMarks,
    AttendanceSummaryOut,
    SessionChange,
    SessionCreate,
    SessionOut,
    SheetOut,
    SlotCreate,
    SlotOut,
)

router = APIRouter(tags=["schedule"])

CohortId = Annotated[int, Query(description="The cohort (the demo has one: 1)")]
From = Annotated[date | None, Query(alias="from", description="First day (default: 30 days before `to`)")]
To = Annotated[date | None, Query(alias="to", description="Last day (default: today)")]


# --- the template --------------------------------------------------------------------------------


@router.get("/schedule/slots", summary="The weekly template, Sunday first")
def list_slots(actor: Actor, service: ScheduleServiceDep, cohort_id: CohortId) -> list[SlotOut]:
    return [SlotOut.of(s) for s in service.slots(cohort_id)]


@router.post("/schedule/slots", status_code=status.HTTP_201_CREATED, summary="Add a slot to the template (manager)")
def add_slot(body: SlotCreate, actor: Actor, service: ScheduleServiceDep) -> SlotOut:
    return SlotOut.of(service.add_slot(actor, ScheduleSlot(**body.model_dump())))


@router.post(
    "/schedule/slots/{slot_id}/deactivate",
    summary="Stop publishing a slot (manager). The weeks already made from it stay",
)
def deactivate_slot(slot_id: int, actor: Actor, service: ScheduleServiceDep) -> SlotOut:
    return SlotOut.of(service.deactivate_slot(actor, slot_id))


# --- weeks ---------------------------------------------------------------------------------------


@router.post(
    "/schedule/weeks/{sunday}/publish",
    summary="Create a week's sessions from the template (manager)",
    description="The week starts on Sunday. Safe to run again: only slots without a session that week are added.",
)
def publish_week(
    sunday: date, actor: Actor, service: ScheduleServiceDep, cohort_id: CohortId
) -> list[SessionOut]:
    return [SessionOut.of(s) for s in service.publish_week(actor, cohort_id, sunday)]


@router.get(
    "/schedule/weeks/{day}",
    summary="A week's sessions: give any day of it",
    description="Cancelled sessions are included, marked cancelled.",
)
def get_week(day: date, actor: Actor, service: ScheduleServiceDep, cohort_id: CohortId) -> list[SessionOut]:
    return [SessionOut.of(s) for s in service.week(cohort_id, week_start(day))]


@router.post("/schedule/sessions", status_code=status.HTTP_201_CREATED, summary="A one-off session (manager)")
def add_session(body: SessionCreate, actor: Actor, service: ScheduleServiceDep) -> SessionOut:
    return SessionOut.of(service.add_session(actor, ScheduledSession(**body.model_dump())))


@router.patch("/schedule/sessions/{session_id}", summary="Change one session of one week (manager)")
def change_session(session_id: int, body: SessionChange, actor: Actor, service: ScheduleServiceDep) -> SessionOut:
    return SessionOut.of(service.change_session(actor, session_id, **body.model_dump(exclude_none=True)))


@router.post(
    "/schedule/sessions/{session_id}/cancel",
    summary="Cancel a session (manager): no attendance, and not counted in any rate",
)
def cancel_session(session_id: int, actor: Actor, service: ScheduleServiceDep) -> SessionOut:
    return SessionOut.of(service.cancel_session(actor, session_id))


# --- attendance ----------------------------------------------------------------------------------


@router.get("/schedule/sessions/{session_id}/attendance", summary="The attendance sheet: the trainees you mark here")
def get_sheet(session_id: int, actor: Actor, service: AttendanceServiceDep) -> SheetOut:
    return SheetOut.of(*service.sheet(actor, session_id))


@router.put(
    "/schedule/sessions/{session_id}/attendance",
    summary="Mark attendance: present, absent or excused",
    description="All marks or none. Marking a trainee again replaces the mark. From the session's day on; "
    "never on a cancelled session. No reason text, on purpose: a reason is often health data.",
)
def mark_attendance(
    session_id: int, body: AttendanceMarks, actor: Actor, service: AttendanceServiceDep, today: TodayDep
) -> SheetOut:
    service.mark(actor, session_id, [Mark(m.trainee_id, m.status) for m in body.marks], today=today)
    return SheetOut.of(*service.sheet(actor, session_id))


def _range(first: date | None, last: date | None, today: date) -> tuple[date, date]:
    """The default: the last 30 days, up to today."""
    last = last or today
    return first or last - timedelta(days=29), last


@router.get(
    "/attendance",
    summary="Attendance rates of the trainees you may see, the lowest first",
    description="rate = present / (present + absent). Excused and unmarked are shown, and not counted. "
    "The default range is the last 30 days.",
)
def attendance_summary(
    actor: Actor,
    service: AttendanceServiceDep,
    today: TodayDep,
    cohort_id: CohortId,
    group_id: int | None = None,
    first: From = None,
    last: To = None,
) -> list[AttendanceSummaryOut]:
    start, end = _range(first, last, today)
    rows = service.group_summary(actor, cohort_id, start, end, today=today, group_id=group_id)
    return [AttendanceSummaryOut.of(r) for r in rows]


@router.get("/trainees/{trainee_id}/attendance", summary="One trainee's attendance rate (default: the last 30 days)")
def trainee_attendance(
    trainee_id: int,
    actor: Actor,
    service: AttendanceServiceDep,
    today: TodayDep,
    first: From = None,
    last: To = None,
) -> AttendanceSummaryOut:
    start, end = _range(first, last, today)
    return AttendanceSummaryOut.of(service.trainee_summary(actor, trainee_id, start, end, today=today))
