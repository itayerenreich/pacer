"""The schedule and attendance against PostgreSQL: the constraints hold, marking again replaces,
and the demo's schedule loads."""

from datetime import date, time, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from pacer.demo.__main__ import load
from pacer.domain import AttendanceRecord, AttendanceStatus, ScheduledSession, ScheduleSlot, Weekday
from pacer.repositories import repositories

pytestmark = pytest.mark.integration


@pytest.fixture
def academy(engine):
    return load(engine, reset=False, seed=2026)


def test_the_demo_schedule_loads(engine, academy):
    with repositories(engine) as repos:
        slots = repos.slots.list_by_cohort(1)
        week = repos.sessions.list_between(1, date(2026, 9, 1), date(2027, 6, 30))
    assert len(slots) == academy.slot_count > 0
    assert len(week) == academy.session_count > 0
    with engine.connect() as connection:
        assert connection.execute(text("SELECT count(*) FROM attendance")).scalar() == academy.attendance_count


def test_publishing_the_same_slot_twice_on_one_date_is_refused_by_the_database(engine, academy):
    with repositories(engine) as repos:
        slot = repos.slots.list_by_cohort(1)[0]
        existing = repos.sessions.list_between(1, date(2026, 9, 1), date(2027, 6, 30))
        sunday = next(s.on for s in existing if s.slot_id == slot.id) - timedelta(days=slot.weekday.offset)
    with pytest.raises(IntegrityError, match="uq_scheduled_sessions_slot_date"), repositories(engine) as repos:
        repos.sessions.add(ScheduledSession.from_slot(slot, sunday))


def test_marking_again_replaces_and_a_session_takes_its_marks_with_it(engine, academy):
    with repositories(engine) as repos:
        slot = repos.slots.add(
            ScheduleSlot(cohort_id=1, weekday=Weekday.SUNDAY, start=time(22), end=time(23), title="בדיקה")
        )
        session = repos.sessions.add(ScheduledSession.from_slot(slot, date(2026, 10, 18)))
        repos.attendance.save(AttendanceRecord(session_id=session.key, trainee_id=1, status=AttendanceStatus.ABSENT, recorded_by_id=2))
        repos.attendance.save(AttendanceRecord(session_id=session.key, trainee_id=1, status=AttendanceStatus.EXCUSED, recorded_by_id=3))
    with repositories(engine) as repos:
        (record,) = repos.attendance.list_by_session(session.key)
        assert (record.status, record.recorded_by_id) == (AttendanceStatus.EXCUSED, 3)
    with engine.begin() as connection:
        connection.execute(text("DELETE FROM scheduled_sessions WHERE id = :id"), {"id": session.key})
        left = connection.execute(text("SELECT count(*) FROM attendance WHERE session_id = :id"), {"id": session.key})
        assert left.scalar() == 0


def test_the_database_rejects_an_unknown_status_and_backwards_times(engine, academy):
    with pytest.raises(IntegrityError, match="ck_attendance_status"), engine.begin() as connection:
        connection.execute(text("UPDATE attendance SET status = 'late' WHERE id = (SELECT min(id) FROM attendance)"))
    with pytest.raises(IntegrityError, match="ck_schedule_slots_times"), engine.begin() as connection:
        connection.execute(text("UPDATE schedule_slots SET end_time = start_time WHERE id = 1"))


def test_attendance_for_a_range_only(engine, academy):
    with repositories(engine) as repos:
        everything = repos.attendance.list_for_trainees([1], date(2026, 9, 1), date(2027, 6, 30))
        one_day = repos.attendance.list_for_trainees([1], date(2026, 9, 1), date(2026, 9, 1))
    assert everything and one_day == []


def test_two_counselors_marking_the_same_trainee_at_the_same_moment_both_succeed(engine, academy):
    """INSERT ... ON CONFLICT: no read-then-insert race between the group's two counselors."""
    import threading

    with repositories(engine) as repos:
        slot = repos.slots.add(ScheduleSlot(cohort_id=1, weekday=Weekday.SUNDAY, start=time(22), end=time(23), title="בדיקה"))
        session = repos.sessions.add(ScheduledSession.from_slot(slot, date(2026, 10, 18)))
    barrier, errors = threading.Barrier(2), []

    def mark(counselor: int, status: AttendanceStatus) -> None:
        try:
            with repositories(engine) as repos:
                barrier.wait()
                repos.attendance.save(AttendanceRecord(session_id=session.key, trainee_id=1, status=status, recorded_by_id=counselor))
        except Exception as error:  # noqa: BLE001
            errors.append(type(error).__name__)

    threads = [threading.Thread(target=mark, args=a) for a in ((2, AttendanceStatus.PRESENT), (3, AttendanceStatus.ABSENT))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    with repositories(engine) as repos:
        assert len(repos.attendance.list_by_session(session.key)) == 1


def test_a_database_conflict_is_409_with_a_fixed_message_not_500():
    """A rule only the database saw (a race the services cannot): a conflict, and no SQL in the answer."""
    from sqlalchemy.exc import IntegrityError

    from pacer.api.errors import _FIXED_DETAIL, status_for

    error = IntegrityError("INSERT ...", None, Exception("duplicate key"))
    assert status_for(error) == 409
    assert "SQL" not in _FIXED_DETAIL[IntegrityError] and "INSERT" not in _FIXED_DETAIL[IntegrityError]

