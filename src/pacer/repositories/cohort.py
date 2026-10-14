"""Persistence for cohorts and groups (including the group_counselors link table)."""

from sqlalchemy import insert, select
from sqlalchemy.orm import Session

from pacer.domain import Cohort, Group

from ._mapping import row_values
from .tables import CohortRow, GroupRow, group_counselors


class CohortRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, cohort: Cohort) -> Cohort:
        row = CohortRow(**row_values(cohort))
        self._session.add(row)
        self._session.flush()
        return cohort.model_copy(update={"id": row.id})

    def get(self, cohort_id: int) -> Cohort | None:
        row = self._session.get(CohortRow, cohort_id)
        return None if row is None else Cohort.model_validate(row, from_attributes=True)

    def list_all(self) -> list[Cohort]:
        rows = self._session.scalars(select(CohortRow).order_by(CohortRow.start_date))
        return [Cohort.model_validate(row, from_attributes=True) for row in rows]


class GroupRepository:
    """A Group is stored in two tables: groups, and one group_counselors row per counselor."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, group: Group) -> Group:
        row = GroupRow(**row_values(group, exclude={"counselor_ids"}))
        self._session.add(row)
        self._session.flush()
        if group.counselor_ids:
            self._session.execute(
                insert(group_counselors),
                [{"group_id": row.id, "counselor_id": c} for c in group.counselor_ids],
            )
        return group.model_copy(update={"id": row.id})

    def get(self, group_id: int) -> Group | None:
        row = self._session.get(GroupRow, group_id)
        return None if row is None else self._to_domain(row)

    def list_by_cohort(self, cohort_id: int) -> list[Group]:
        rows = self._session.scalars(
            select(GroupRow).where(GroupRow.cohort_id == cohort_id).order_by(GroupRow.id)
        )
        return [self._to_domain(row) for row in rows]

    def _to_domain(self, row: GroupRow) -> Group:
        counselor_ids = self._session.scalars(
            select(group_counselors.c.counselor_id)
            .where(group_counselors.c.group_id == row.id)
            .order_by(group_counselors.c.counselor_id)
        ).all()
        return Group(id=row.id, name=row.name, cohort_id=row.cohort_id, counselor_ids=list(counselor_ids))
