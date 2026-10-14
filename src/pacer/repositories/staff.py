"""Persistence for staff members."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from pacer.domain import StaffMember

from ._mapping import row_values
from .tables import StaffMemberRow


class StaffRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, staff: StaffMember) -> StaffMember:
        row = StaffMemberRow(**row_values(staff))
        self._session.add(row)
        self._session.flush()  # sends the INSERT inside the transaction, so the id is known
        return staff.model_copy(update={"id": row.id})

    def get(self, staff_id: int) -> StaffMember | None:
        row = self._session.get(StaffMemberRow, staff_id)
        return None if row is None else _to_domain(row)

    def list_all(self) -> list[StaffMember]:
        rows = self._session.scalars(select(StaffMemberRow).order_by(StaffMemberRow.id))
        return [_to_domain(row) for row in rows]


def _to_domain(row: StaffMemberRow) -> StaffMember:
    return StaffMember.model_validate(row, from_attributes=True)
