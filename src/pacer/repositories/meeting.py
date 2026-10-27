"""Persistence for staff meetings (with their participants), proposed tasks and tasks."""

from collections.abc import Sequence

from sqlalchemy import delete, insert, select
from sqlalchemy.orm import Session

from pacer.domain import ProposalStatus, ProposedTask, StaffMeeting, Task

from ._mapping import row_values
from .tables import ProposedTaskRow, StaffMeetingRow, TaskRow, meeting_participants


class MeetingRepository:
    """A meeting is stored in two tables: staff_meetings, and one meeting_participants row per person."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, meeting: StaffMeeting) -> StaffMeeting:
        row = StaffMeetingRow(**row_values(meeting, exclude={"participant_ids"}))
        self._session.add(row)
        self._session.flush()
        self._write_participants(row.id, meeting.participant_ids)
        return meeting.model_copy(update={"id": row.id})

    def get(self, meeting_id: int) -> StaffMeeting | None:
        row = self._session.get(StaffMeetingRow, meeting_id)
        return None if row is None else self._to_domain(row)

    def lock(self, meeting_id: int) -> StaffMeeting | None:
        """SELECT ... FOR UPDATE: a second transaction that locks the same meeting waits here until
        the first one commits, and then reads the meeting as the first one left it. This is what
        makes "check the status, then write" safe when two requests process the same meeting."""
        row = self._session.scalars(
            select(StaffMeetingRow)
            .where(StaffMeetingRow.id == meeting_id)
            .with_for_update()
            .execution_options(populate_existing=True)  # re-read, even if this session saw it before
        ).first()
        return None if row is None else self._to_domain(row)

    def update(self, meeting: StaffMeeting) -> StaffMeeting:
        row = self._session.get(StaffMeetingRow, meeting.id)
        if row is None:
            raise LookupError(f"Meeting {meeting.id} does not exist")
        for column, value in row_values(meeting, exclude={"participant_ids"}).items():
            setattr(row, column, value)
        self._session.execute(delete(meeting_participants).where(meeting_participants.c.meeting_id == row.id))
        self._write_participants(row.id, meeting.participant_ids)
        self._session.flush()
        return meeting

    def list_all(self) -> list[StaffMeeting]:
        rows = self._session.scalars(select(StaffMeetingRow).order_by(StaffMeetingRow.held_on.desc()))
        return [self._to_domain(row) for row in rows]

    def _write_participants(self, meeting_id: int, staff_ids: list[int]) -> None:
        self._session.execute(
            insert(meeting_participants), [{"meeting_id": meeting_id, "staff_id": s} for s in staff_ids]
        )

    def _to_domain(self, row: StaffMeetingRow) -> StaffMeeting:
        participant_ids: Sequence[int] = self._session.scalars(
            select(meeting_participants.c.staff_id)
            .where(meeting_participants.c.meeting_id == row.id)
            .order_by(meeting_participants.c.staff_id)
        ).all()
        data = {
            column: getattr(row, column)
            for column in StaffMeeting.model_fields
            if column != "participant_ids"
        }
        return StaffMeeting.model_validate({**data, "participant_ids": list(participant_ids)})


class ProposedTaskRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, proposal: ProposedTask) -> ProposedTask:
        row = ProposedTaskRow(**row_values(proposal))
        self._session.add(row)
        self._session.flush()
        return proposal.model_copy(update={"id": row.id})

    def get(self, proposal_id: int) -> ProposedTask | None:
        row = self._session.get(ProposedTaskRow, proposal_id)
        return None if row is None else ProposedTask.model_validate(row, from_attributes=True)

    def lock(self, proposal_id: int) -> ProposedTask | None:
        """SELECT ... FOR UPDATE, re-read even if this session saw the row before."""
        row = self._session.scalars(
            select(ProposedTaskRow)
            .where(ProposedTaskRow.id == proposal_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        ).first()
        return None if row is None else ProposedTask.model_validate(row, from_attributes=True)

    def update(self, proposal: ProposedTask) -> ProposedTask:
        row = self._session.get(ProposedTaskRow, proposal.id)
        if row is None:
            raise LookupError(f"Proposed task {proposal.id} does not exist")
        for column, value in row_values(proposal).items():
            setattr(row, column, value)
        self._session.flush()
        return proposal

    def list_by_meeting(self, meeting_id: int) -> list[ProposedTask]:
        rows = self._session.scalars(
            select(ProposedTaskRow)
            .where(ProposedTaskRow.meeting_id == meeting_id)
            .order_by(ProposedTaskRow.id)
        )
        return [ProposedTask.model_validate(row, from_attributes=True) for row in rows]

    def count_open(self, meeting_id: int) -> int:
        return len(
            self._session.scalars(
                select(ProposedTaskRow.id).where(
                    ProposedTaskRow.meeting_id == meeting_id,
                    ProposedTaskRow.status == ProposalStatus.PROPOSED.value,
                )
            ).all()
        )


class TaskRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, task: Task) -> Task:
        row = TaskRow(**row_values(task))
        self._session.add(row)
        self._session.flush()
        return task.model_copy(update={"id": row.id})

    def get(self, task_id: int) -> Task | None:
        row = self._session.get(TaskRow, task_id)
        return None if row is None else Task.model_validate(row, from_attributes=True)

    def update(self, task: Task) -> Task:
        row = self._session.get(TaskRow, task.id)
        if row is None:
            raise LookupError(f"Task {task.id} does not exist")
        for column, value in row_values(task).items():
            setattr(row, column, value)
        self._session.flush()
        return task

    def list_by_owner(self, owner_id: int) -> list[Task]:
        rows = self._session.scalars(select(TaskRow).where(TaskRow.owner_id == owner_id).order_by(TaskRow.id))
        return [Task.model_validate(row, from_attributes=True) for row in rows]

    def list_by_trainee(self, trainee_id: int) -> list[Task]:
        rows = self._session.scalars(select(TaskRow).where(TaskRow.trainee_id == trainee_id).order_by(TaskRow.id))
        return [Task.model_validate(row, from_attributes=True) for row in rows]

    def list_all(self) -> list[Task]:
        rows = self._session.scalars(select(TaskRow).order_by(TaskRow.id))
        return [Task.model_validate(row, from_attributes=True) for row in rows]

    def delete(self, task_id: int) -> None:
        row = self._session.get(TaskRow, task_id)
        if row is not None:
            self._session.delete(row)
            self._session.flush()
