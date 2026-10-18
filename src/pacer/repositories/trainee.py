"""Persistence for trainees and their sensitive personal files."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from pacer.domain import PersonalFile, Trainee

from ._mapping import row_values
from .tables import PersonalFileRow, TraineeRow


class TraineeRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, trainee: Trainee) -> Trainee:
        row = TraineeRow(**row_values(trainee))
        self._session.add(row)
        self._session.flush()
        return trainee.model_copy(update={"id": row.id})

    def get(self, trainee_id: int) -> Trainee | None:
        row = self._session.get(TraineeRow, trainee_id)
        return None if row is None else _to_domain(row)

    def update(self, trainee: Trainee) -> Trainee:
        row = self._session.get(TraineeRow, trainee.id)
        if row is None:
            raise LookupError(f"Trainee {trainee.id} does not exist")
        for column, value in row_values(trainee).items():
            setattr(row, column, value)
        self._session.flush()
        return trainee

    def lock(self, trainee_id: int) -> Trainee | None:
        """SELECT ... FOR UPDATE: the row stays locked until the unit of work ends."""
        row = self._session.get(TraineeRow, trainee_id, with_for_update=True, populate_existing=True)
        return None if row is None else _to_domain(row)

    def list_all(self) -> list[Trainee]:
        rows = self._session.scalars(select(TraineeRow).order_by(TraineeRow.id))
        return [_to_domain(row) for row in rows]

    def list_by_group(self, group_id: int) -> list[Trainee]:
        rows = self._session.scalars(
            select(TraineeRow).where(TraineeRow.group_id == group_id).order_by(TraineeRow.id)
        )
        return [_to_domain(row) for row in rows]

    def list_by_counselor(self, counselor_id: int) -> list[Trainee]:
        rows = self._session.scalars(
            select(TraineeRow)
            .where(TraineeRow.primary_counselor_id == counselor_id)
            .order_by(TraineeRow.id)
        )
        return [_to_domain(row) for row in rows]


class PersonalFileRepository:
    """Sensitive data. In week 2 every access goes through a permission check first."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, personal_file: PersonalFile) -> PersonalFile:
        row = PersonalFileRow(**row_values(personal_file))
        self._session.add(row)
        self._session.flush()
        return personal_file.model_copy(update={"id": row.id})

    def get_by_trainee(self, trainee_id: int) -> PersonalFile | None:
        row = self._session.scalars(
            select(PersonalFileRow).where(PersonalFileRow.trainee_id == trainee_id)
        ).one_or_none()
        return None if row is None else PersonalFile.model_validate(row, from_attributes=True)


def _to_domain(row: TraineeRow) -> Trainee:
    return Trainee.model_validate(row, from_attributes=True)
