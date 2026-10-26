"""The weekly schedule and attendance: the template, weeks, who marks what, and the rates."""

from datetime import date, time

import pytest
from fakes import World

from pacer.domain import (
    AttendanceStatus,
    DomainRuleError,
    Group,
    ScheduledSession,
    ScheduleSlot,
    TraineeStatus,
    Weekday,
    week_start,
)
from pacer.services import (
    AttendanceService,
    Mark,
    NotFoundError,
    PermissionDeniedError,
    RuleViolationError,
    ScheduleService,
)

P, A, E = AttendanceStatus.PRESENT, AttendanceStatus.ABSENT, AttendanceStatus.EXCUSED
SUNDAY = date(2026, 10, 18)  # a Sunday inside the cohort (2026-09-01 to 2027-06-30)
TODAY = date(2026, 10, 25)


@pytest.fixture
def world() -> World:
    w = World()
    # A second trainee in group A, and one in group B (Dana's), so the rules have something to tell apart.
    w.ori = w.trainees.add(w.yael.model_copy(update={"id": None, "first_name": "Ori", "last_name": "Bar"}))
    w.tom = w.trainees.add(
        w.yael.model_copy(update={"id": None, "first_name": "Tom", "group_id": w.group_b.id, "primary_counselor_id": w.dana.id})
    )
    return w


@pytest.fixture
def schedule(world) -> ScheduleService:
    return ScheduleService(world.slots, world.sessions, world.cohorts, world.groups, world.attendance)


@pytest.fixture
def attendance(world) -> AttendanceService:
    return AttendanceService(world.sessions, world.attendance, world.trainees, world.groups)


def slot(world, weekday=Weekday.SUNDAY, start=time(7), end=time(8), title="ספורט בוקר", group_id=None) -> ScheduleSlot:
    return ScheduleSlot(cohort_id=world.cohort.key, group_id=group_id, weekday=weekday, start=start, end=end, title=title)


def published(world, schedule, *slots: ScheduleSlot, sunday=SUNDAY) -> list[ScheduledSession]:
    for s in slots:
        schedule.add_slot(world.manager, s)
    return schedule.publish_week(world.manager, world.cohort.key, sunday)


# --- the domain ----------------------------------------------------------------------------------


def test_the_academy_week_starts_on_sunday():
    assert week_start(date(2026, 10, 22)) == SUNDAY  # Thursday
    assert week_start(SUNDAY) == SUNDAY
    assert week_start(date(2026, 10, 24)) == SUNDAY  # Saturday ends the week
    assert [d.offset for d in (Weekday.SUNDAY, Weekday.MONDAY, Weekday.SATURDAY)] == [0, 1, 6]


def test_a_slot_ends_after_it_starts(world):
    with pytest.raises(ValueError):
        slot(world, start=time(8), end=time(8))


def test_a_cancelled_session_cannot_be_changed_or_cancelled_again(world, schedule):
    (session,) = published(world, schedule, slot(world))
    cancelled = session.cancel()
    with pytest.raises(DomainRuleError):
        cancelled.changed(title="אחר")
    with pytest.raises(DomainRuleError):
        cancelled.cancel()


# --- the template and weeks ----------------------------------------------------------------------


def test_publishing_a_week_puts_each_slot_on_its_day(world, schedule):
    sessions = published(world, schedule, slot(world, Weekday.SUNDAY), slot(world, Weekday.THURSDAY, title="שיעור עיון"))
    assert [(s.on, s.title) for s in sessions] == [(SUNDAY, "ספורט בוקר"), (date(2026, 10, 22), "שיעור עיון")]
    assert all(s.slot_id is not None for s in sessions)


def test_publishing_again_adds_only_what_is_new(world, schedule):
    published(world, schedule, slot(world))
    schedule.add_slot(world.manager, slot(world, Weekday.MONDAY, title="סדנה"))
    again = schedule.publish_week(world.manager, world.cohort.key, SUNDAY)
    assert [s.title for s in again] == ["סדנה"]
    assert len(schedule.week(world.cohort.key, SUNDAY)) == 2


def test_a_week_starts_on_sunday_and_is_inside_the_cohort(world, schedule):
    schedule.add_slot(world.manager, slot(world))
    with pytest.raises(RuleViolationError, match="Sunday"):
        schedule.publish_week(world.manager, world.cohort.key, date(2026, 10, 19))
    with pytest.raises(RuleViolationError, match="outside the cohort"):
        schedule.publish_week(world.manager, world.cohort.key, date(2027, 8, 1))


def test_an_inactive_slot_is_not_published_and_its_past_weeks_stay(world, schedule):
    (past,) = published(world, schedule, slot(world))
    schedule.deactivate_slot(world.manager, past.slot_id)
    assert schedule.publish_week(world.manager, world.cohort.key, date(2026, 10, 25)) == []
    assert world.sessions.get(past.key) == past


def test_changing_one_week_does_not_touch_the_template(world, schedule):
    (session,) = published(world, schedule, slot(world))
    moved = schedule.change_session(world.manager, session.key, start=time(9), end=time(10))
    assert moved.start == time(9)
    assert world.slots.get(session.slot_id).start == time(7)


def test_a_group_slot_must_be_a_group_of_the_cohort(world, schedule):
    other = world.groups.add(Group(name="Elsewhere", cohort_id=999, counselor_ids=[]))
    with pytest.raises(RuleViolationError):
        schedule.add_slot(world.manager, slot(world, group_id=other.key))


def test_only_the_manager_changes_the_schedule(world, schedule):
    with pytest.raises(PermissionDeniedError):
        schedule.add_slot(world.noa, slot(world))
    (session,) = published(world, schedule, slot(world))
    for change in (
        lambda: schedule.publish_week(world.noa, world.cohort.key, SUNDAY),
        lambda: schedule.cancel_session(world.noa, session.key),
        lambda: schedule.change_session(world.noa, session.key, title="אחר"),
        lambda: schedule.deactivate_slot(world.noa, session.slot_id),
    ):
        with pytest.raises(PermissionDeniedError):
            change()


def test_a_one_off_session_has_no_template_slot(world, schedule):
    trip = ScheduledSession(cohort_id=world.cohort.key, on=SUNDAY, start=time(8), end=time(18), title="טיול שנתי")
    assert schedule.add_session(world.manager, trip).slot_id is None
    with pytest.raises(RuleViolationError):
        schedule.add_session(world.manager, trip.model_copy(update={"slot_id": 1}))


# --- marking -------------------------------------------------------------------------------------


def test_the_sheet_shows_the_trainees_this_actor_marks(world, schedule, attendance):
    (session,) = published(world, schedule, slot(world))
    _, mine = attendance.sheet(world.noa, session.key)
    assert [r.trainee.first_name for r in mine] == ["Yael", "Ori"]  # group A only
    _, everyone = attendance.sheet(world.manager, session.key)
    assert len(everyone) == 3 and all(r.status is None for r in everyone)


def test_a_group_session_expects_only_that_group(world, schedule, attendance):
    (talk,) = published(world, schedule, slot(world, title="שיחת קבוצה", group_id=world.group_b.key))
    _, rows = attendance.sheet(world.manager, talk.key)
    assert [r.trainee.first_name for r in rows] == ["Tom"]
    with pytest.raises(RuleViolationError, match="not expected"):
        attendance.mark(world.manager, talk.key, [Mark(world.yael.key, P)], today=TODAY)


def test_a_trainee_who_is_not_active_is_not_expected(world, schedule, attendance):
    world.trainees.update(world.ori.with_status(TraineeStatus.ON_LEAVE))
    (session,) = published(world, schedule, slot(world))
    _, rows = attendance.sheet(world.noa, session.key)
    assert [r.trainee.first_name for r in rows] == ["Yael"]


def test_marking_again_replaces_the_mark(world, schedule, attendance):
    (session,) = published(world, schedule, slot(world))
    attendance.mark(world.noa, session.key, [Mark(world.yael.key, A)], today=TODAY)
    attendance.mark(world.omer, session.key, [Mark(world.yael.key, E)], today=TODAY)
    (record,) = world.attendance.list_by_session(session.key)
    assert record.status is E and record.recorded_by_id == world.omer.key


def test_a_counselor_cannot_mark_a_trainee_of_another_group_and_learns_nothing(world, schedule, attendance):
    (session,) = published(world, schedule, slot(world))
    with pytest.raises(NotFoundError):
        attendance.mark(world.noa, session.key, [Mark(world.tom.key, P)], today=TODAY)
    assert world.attendance.list_by_session(session.key) == []


def test_all_marks_or_none(world, schedule, attendance):
    """One bad mark stops the request; the unit of work rolls back what came before it."""
    (session,) = published(world, schedule, slot(world))
    with pytest.raises(NotFoundError), world.uow():
        attendance.mark(world.noa, session.key, [Mark(world.yael.key, P), Mark(world.tom.key, P)], today=TODAY)
    assert world.attendance.list_by_session(session.key) == []


def test_no_marking_before_the_day_or_on_a_cancelled_session(world, schedule, attendance):
    (future,) = published(world, schedule, slot(world), sunday=date(2026, 11, 1))
    with pytest.raises(RuleViolationError, match="from that day on"):
        attendance.mark(world.noa, future.key, [Mark(world.yael.key, P)], today=TODAY)
    cancelled = schedule.cancel_session(world.manager, future.key)
    with pytest.raises(RuleViolationError, match="cancelled"):
        attendance.mark(world.noa, cancelled.key, [Mark(world.yael.key, P)], today=date(2026, 11, 2))


def test_a_trainee_twice_in_one_request_is_refused(world, schedule, attendance):
    (session,) = published(world, schedule, slot(world))
    with pytest.raises(RuleViolationError, match="twice"):
        attendance.mark(world.noa, session.key, [Mark(world.yael.key, P), Mark(world.yael.key, A)], today=TODAY)


# --- rates ---------------------------------------------------------------------------------------


@pytest.fixture
def four_sessions(world, schedule, attendance):
    """Sunday to Wednesday, one session a day. Yael: present, absent, excused, and one unmarked."""
    sessions = published(
        world,
        schedule,
        *(slot(world, day, title=t) for day, t in [
            (Weekday.SUNDAY, "ספורט בוקר"), (Weekday.MONDAY, "ספורט בוקר"), (Weekday.TUESDAY, "שיעור"), (Weekday.WEDNESDAY, "שיעור"),
        ]),
    )
    for session, status in zip(sessions[:3], (P, A, E), strict=True):
        attendance.mark(world.noa, session.key, [Mark(world.yael.key, status)], today=TODAY)
    return sessions


def test_the_rate_is_present_out_of_present_and_absent(world, attendance, four_sessions):
    summary = attendance.trainee_summary(world.noa, world.yael.key, SUNDAY, TODAY, today=TODAY)
    assert (summary.present, summary.absent, summary.excused, summary.unmarked) == (1, 1, 1, 1)
    assert summary.rate == 0.5  # excused and unmarked do not count
    assert summary.missed_by_title == {"ספורט בוקר": 1}


def test_a_future_session_is_not_unmarked(world, attendance, four_sessions):
    summary = attendance.trainee_summary(world.noa, world.yael.key, SUNDAY, date(2026, 10, 31), today=date(2026, 10, 20))
    assert summary.unmarked == 0  # Wednesday has not come yet
    assert (summary.present, summary.absent, summary.excused) == (1, 1, 1)


def test_a_cancelled_session_leaves_the_rate_even_if_it_was_marked(world, schedule, attendance, four_sessions):
    schedule.cancel_session(world.manager, four_sessions[1].key)  # the absence
    summary = attendance.trainee_summary(world.noa, world.yael.key, SUNDAY, TODAY, today=TODAY)
    assert (summary.present, summary.absent, summary.rate) == (1, 0, 1.0)


def test_nothing_counted_yet_is_no_rate_not_zero(world, attendance, four_sessions):
    summary = attendance.trainee_summary(world.noa, world.ori.key, SUNDAY, TODAY, today=TODAY)
    assert summary.rate is None and summary.unmarked == 4


def test_the_group_summary_puts_the_lowest_rate_first_and_hides_other_groups(world, attendance, four_sessions):
    attendance.mark(world.noa, four_sessions[0].key, [Mark(world.ori.key, P)], today=TODAY)
    rows = attendance.group_summary(world.noa, world.cohort.key, SUNDAY, TODAY, today=TODAY)
    assert [(r.trainee.first_name, r.rate) for r in rows] == [("Yael", 0.5), ("Ori", 1.0)]
    everyone = attendance.group_summary(world.manager, world.cohort.key, SUNDAY, TODAY, today=TODAY)
    assert [r.trainee.first_name for r in everyone] == ["Yael", "Ori", "Tom"]  # Tom: nothing counted, last


def test_a_group_the_actor_does_not_lead_is_404(world, attendance):
    with pytest.raises(NotFoundError):
        attendance.group_summary(world.noa, world.cohort.key, SUNDAY, TODAY, today=TODAY, group_id=world.group_b.key)


def test_a_trainee_of_another_group_is_404_for_the_summary(world, attendance):
    with pytest.raises(NotFoundError):
        attendance.trainee_summary(world.noa, world.tom.key, SUNDAY, TODAY, today=TODAY)


def test_a_range_that_ends_before_it_starts_is_refused(world, attendance):
    with pytest.raises(RuleViolationError):
        attendance.trainee_summary(world.noa, world.yael.key, TODAY, SUNDAY, today=TODAY)


def test_group_sessions_count_only_for_their_group(world, schedule, attendance):
    published(world, schedule, slot(world, title="שיחת קבוצה", group_id=world.group_b.key))
    summary = attendance.trainee_summary(world.manager, world.yael.key, SUNDAY, TODAY, today=TODAY)
    assert summary.unmarked == 0  # group B's talk is not Yael's


# --- the independent review's findings (2026-10-26) ----------------------------------------------


def test_publishing_again_after_moving_a_session_in_the_week_adds_no_duplicate(world, schedule):
    (session,) = published(world, schedule, slot(world))
    schedule.change_session(world.manager, session.key, on=date(2026, 10, 19))  # Sunday -> Monday
    assert schedule.publish_week(world.manager, world.cohort.key, SUNDAY) == []
    assert len(schedule.week(world.cohort.key, SUNDAY)) == 1


def test_a_template_session_moves_only_within_its_week(world, schedule):
    (session,) = published(world, schedule, slot(world))
    with pytest.raises(RuleViolationError, match="own week"):
        schedule.change_session(world.manager, session.key, on=date(2026, 10, 26))


def test_a_template_session_cannot_land_on_another_session_of_its_slot(world, schedule):
    """The database's unique (slot, date) would be a 500; the service says it first, as a 409. (Two
    sessions of one slot in a week happen when the manager adds the slot twice by mistake.)"""
    (sunday,) = published(world, schedule, slot(world))
    twin = world.sessions.add(sunday.model_copy(update={"id": None, "on": date(2026, 10, 20)}))
    with pytest.raises(RuleViolationError, match="already has a session"):
        schedule.change_session(world.manager, sunday.key, on=twin.on)
    assert world.sessions.get(sunday.key).on == SUNDAY


def test_a_session_stays_inside_the_cohort(world, schedule):
    trip = ScheduledSession(cohort_id=world.cohort.key, on=date(2027, 7, 15), start=time(8), end=time(18), title="טיול")
    with pytest.raises(RuleViolationError, match="outside the cohort"):
        schedule.add_session(world.manager, trip)
    one_off = schedule.add_session(world.manager, trip.model_copy(update={"on": date(2027, 6, 1)}))
    with pytest.raises(RuleViolationError, match="outside the cohort"):
        schedule.change_session(world.manager, one_off.key, on=date(2027, 7, 1))


def test_a_session_with_attendance_keeps_its_date(world, schedule, attendance):
    (session,) = published(world, schedule, slot(world))
    attendance.mark(world.noa, session.key, [Mark(world.yael.key, P)], today=TODAY)
    with pytest.raises(RuleViolationError, match="has attendance"):
        schedule.change_session(world.manager, session.key, on=date(2026, 10, 19))
    assert schedule.change_session(world.manager, session.key, title="ספורט").title == "ספורט"  # other changes: fine


def test_a_trainee_who_moved_groups_keeps_their_marked_past(world, schedule, attendance):
    (talk,) = published(world, schedule, slot(world, title="שיחת קבוצה", group_id=world.group_a.key))
    attendance.mark(world.noa, talk.key, [Mark(world.yael.key, A)], today=TODAY)
    world.trainees.update(world.yael.model_copy(update={"group_id": world.group_b.key, "primary_counselor_id": world.dana.key}))
    summary = attendance.trainee_summary(world.manager, world.yael.key, SUNDAY, TODAY, today=TODAY)
    assert (summary.absent, summary.rate) == (1, 0.0)
    _, rows = attendance.sheet(world.manager, talk.key)
    assert [(r.trainee.first_name, r.status) for r in rows] == [("Yael", A), ("Ori", None)]


def test_a_trainee_on_leave_stays_on_the_past_sheet(world, schedule, attendance):
    (session,) = published(world, schedule, slot(world))
    attendance.mark(world.noa, session.key, [Mark(world.ori.key, P)], today=TODAY)
    world.trainees.update(world.trainees.get(world.ori.key).with_status(TraineeStatus.ON_LEAVE))
    _, rows = attendance.sheet(world.noa, session.key)
    assert ("Ori", P) in [(r.trainee.first_name, r.status) for r in rows]


def test_todays_sessions_are_not_unmarked_yet(world, schedule, attendance):
    published(world, schedule, slot(world, Weekday.SUNDAY), slot(world, Weekday.MONDAY, title="ערב"))
    summary = attendance.trainee_summary(world.noa, world.yael.key, SUNDAY, date(2026, 10, 19), today=date(2026, 10, 19))
    assert summary.unmarked == 1  # Sunday; Monday's evening is today


def test_a_group_of_another_cohort_is_refused(world, attendance):
    other = world.groups.add(Group(name="Elsewhere", cohort_id=999, counselor_ids=[]))
    with pytest.raises(RuleViolationError):
        attendance.group_summary(world.manager, world.cohort.key, SUNDAY, TODAY, today=TODAY, group_id=other.key)


def test_an_unknown_cohort_is_404_for_the_template_too(schedule):
    with pytest.raises(NotFoundError):
        schedule.slots(999)
