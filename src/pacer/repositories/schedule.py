"""Persistence for the weekly schedule (template slots and dated sessions) and attendance."""

from collections.abc import Iterable
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from pacer.domain import AttendanceRecord, ScheduledSession, ScheduleSlot

from ._mapping import row_values
from .tables import AttendanceRow, ScheduledSessionRow, ScheduleSlotRow


class ScheduleSlotRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, slot: ScheduleSlot) -> ScheduleSlot:
        row = ScheduleSlotRow(**row_values(slot))
        self._session.add(row)
        self._session.flush()
        return slot.model_copy(update={"id": row.id})

    def get(self, slot_id: int) -> ScheduleSlot | None:
        row = self._session.get(ScheduleSlotRow, slot_id)
        return None if row is None else ScheduleSlot.model_validate(row, from_attributes=True)

    def update(self, slot: ScheduleSlot) -> ScheduleSlot:
        row = self._session.get(ScheduleSlotRow, slot.id)
        if row is None:
            raise LookupError(f"Slot {slot.id} does not exist")
        for column, value in row_values(slot).items():
            setattr(row, column, value)
        self._session.flush()
        return slot

    def list_by_cohort(self, cohort_id: int) -> list[ScheduleSlot]:
        rows = self._session.scalars(
            select(ScheduleSlotRow)
            .where(ScheduleSlotRow.cohort_id == cohort_id)
            .order_by(ScheduleSlotRow.weekday, ScheduleSlotRow.start, ScheduleSlotRow.id)
        )
        return [ScheduleSlot.model_validate(row, from_attributes=True) for row in rows]


class ScheduledSessionRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, scheduled: ScheduledSession) -> ScheduledSession:
        row = ScheduledSessionRow(**row_values(scheduled))
        self._session.add(row)
        self._session.flush()
        return scheduled.model_copy(update={"id": row.id})

    def get(self, session_id: int) -> ScheduledSession | None:
        row = self._session.get(ScheduledSessionRow, session_id)
        return None if row is None else ScheduledSession.model_validate(row, from_attributes=True)

    def update(self, scheduled: ScheduledSession) -> ScheduledSession:
        row = self._session.get(ScheduledSessionRow, scheduled.id)
        if row is None:
            raise LookupError(f"Session {scheduled.id} does not exist")
        for column, value in row_values(scheduled).items():
            setattr(row, column, value)
        self._session.flush()
        return scheduled

    def list_between(self, cohort_id: int, first: date, last: date) -> list[ScheduledSession]:
        """Every session of the cohort from `first` to `last`, both included, cancelled ones too."""
        rows = self._session.scalars(
            select(ScheduledSessionRow)
            .where(
                ScheduledSessionRow.cohort_id == cohort_id,
                ScheduledSessionRow.on >= first,
                ScheduledSessionRow.on <= last,
            )
            .order_by(ScheduledSessionRow.on, ScheduledSessionRow.start, ScheduledSessionRow.id)
        )
        return [ScheduledSession.model_validate(row, from_attributes=True) for row in rows]

    def slot_dates(self, slot_ids: Iterable[int], first: date, last: date) -> set[tuple[int, date]]:
        """(slot_id, date) of the sessions already made from these slots in this range."""
        ids = list(slot_ids)
        if not ids:
            return set()
        rows = self._session.execute(
            select(ScheduledSessionRow.slot_id, ScheduledSessionRow.on).where(
                ScheduledSessionRow.slot_id.in_(ids),
                ScheduledSessionRow.on >= first,
                ScheduledSessionRow.on <= last,
            )
        )
        return {(slot_id, on) for slot_id, on in rows if slot_id is not None}


class AttendanceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def save(self, record: AttendanceRecord) -> AttendanceRecord:
        """Insert, or replace the record of the same (session, trainee): marking again corrects it."""
        row = self._session.scalars(
            select(AttendanceRow).where(
                AttendanceRow.session_id == record.session_id, AttendanceRow.trainee_id == record.trainee_id
            )
        ).first()
        if row is None:
            row = AttendanceRow(**row_values(record))
            self._session.add(row)
        else:
            for column, value in row_values(record).items():
                setattr(row, column, value)
        self._session.flush()
        return record.model_copy(update={"id": row.id})

    def list_by_session(self, session_id: int) -> list[AttendanceRecord]:
        rows = self._session.scalars(select(AttendanceRow).where(AttendanceRow.session_id == session_id))
        return [AttendanceRecord.model_validate(row, from_attributes=True) for row in rows]

    def list_for_trainees(
        self, trainee_ids: Iterable[int], first: date, last: date
    ) -> list[AttendanceRecord]:
        """The records of these trainees in sessions from `first` to `last`."""
        ids = list(trainee_ids)
        if not ids:
            return []
        rows = self._session.scalars(
            select(AttendanceRow)
            .join(ScheduledSessionRow, ScheduledSessionRow.id == AttendanceRow.session_id)
            .where(
                AttendanceRow.trainee_id.in_(ids),
                ScheduledSessionRow.on >= first,
                ScheduledSessionRow.on <= last,
            )
        )
        return [AttendanceRecord.model_validate(row, from_attributes=True) for row in rows]
