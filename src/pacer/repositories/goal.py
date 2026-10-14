"""Persistence for goals and their progress history."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from pacer.domain import Goal, GoalUpdate

from ._mapping import row_values
from .tables import GoalRow, GoalUpdateRow


class GoalRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, goal: Goal) -> Goal:
        row = GoalRow(**row_values(goal))
        self._session.add(row)
        self._session.flush()
        return goal.model_copy(update={"id": row.id})

    def get(self, goal_id: int) -> Goal | None:
        row = self._session.get(GoalRow, goal_id)
        return None if row is None else _to_domain(row)

    def update(self, goal: Goal) -> Goal:
        row = self._session.get(GoalRow, goal.id)
        if row is None:
            raise LookupError(f"Goal {goal.id} does not exist")
        for column, value in row_values(goal).items():
            setattr(row, column, value)
        self._session.flush()
        return goal

    def list_by_trainee(self, trainee_id: int) -> list[Goal]:
        rows = self._session.scalars(
            select(GoalRow).where(GoalRow.trainee_id == trainee_id).order_by(GoalRow.id)
        )
        return [_to_domain(row) for row in rows]


class GoalUpdateRepository:
    """History is append-only: updates are added and read, never changed."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, update: GoalUpdate) -> GoalUpdate:
        row = GoalUpdateRow(**row_values(update))
        self._session.add(row)
        self._session.flush()
        return update.model_copy(update={"id": row.id})

    def list_by_goal(self, goal_id: int) -> list[GoalUpdate]:
        rows = self._session.scalars(
            select(GoalUpdateRow)
            .where(GoalUpdateRow.goal_id == goal_id)
            .order_by(GoalUpdateRow.created_at)
        )
        return [GoalUpdate.model_validate(row, from_attributes=True) for row in rows]


def _to_domain(row: GoalRow) -> Goal:
    return Goal.model_validate(row, from_attributes=True)
