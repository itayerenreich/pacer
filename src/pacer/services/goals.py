"""Goals and progress: adding goals, updating progress, reading the history."""

import logging

from pacer.domain import Goal, GoalUpdate, StaffMember, Trainee
from pacer.repositories.interfaces import (
    GoalRepo,
    GoalUpdateRepo,
    GroupRepo,
    TraineeRepo,
)

from .errors import RuleViolationError, found
from .permissions import can_edit_goals, require
from .visibility import visible_trainee

log = logging.getLogger(__name__)


class GoalService:
    """Every method gets the acting staff member first, checks the rules, then acts.

    The repositories are passed in (dependency injection). In production they are
    the SQLAlchemy repositories inside one unit_of_work; in unit tests they are
    in-memory fakes. The service does not know or care which.
    """

    def __init__(
        self, goals: GoalRepo, updates: GoalUpdateRepo, trainees: TraineeRepo, groups: GroupRepo
    ) -> None:
        self._goals = goals
        self._updates = updates
        self._trainees = trainees
        self._groups = groups

    def add_goal(self, actor: StaffMember, goal: Goal) -> Goal:
        trainee = self._open_trainee(actor, goal.trainee_id)
        require(can_edit_goals(actor, trainee), "add goals for this trainee")
        added = self._goals.add(goal)
        log.info("Goal %s added for trainee %s by staff %s", added.id, trainee.id, actor.id)
        return added

    def update_progress(self, actor: StaffMember, goal_id: int, progress: int, note: str) -> Goal:
        """Move a goal to a new progress value, and record who, when, and why.

        The goal row holds the CURRENT state; goal_updates holds every step that led
        there. Both are written in the caller's unit of work, so either both are
        saved or neither is.
        """
        goal, _ = self._record(actor, goal_id, progress, note)
        return goal

    def record_progress(self, actor: StaffMember, goal_id: int, progress: int, note: str) -> GoalUpdate:
        """The same as update_progress, for callers that need the history entry it created
        (an approved talk proposal points to its GoalUpdate)."""
        _, update = self._record(actor, goal_id, progress, note)
        return update

    def _record(self, actor: StaffMember, goal_id: int, progress: int, note: str) -> tuple[Goal, GoalUpdate]:
        goal = found(self._goals.get(goal_id), "Goal", goal_id)
        trainee = self._open_trainee(actor, goal.trainee_id, what="Goal", object_id=goal_id)
        require(can_edit_goals(actor, trainee), "update this goal")

        updated = goal.with_progress(progress)  # a new object; raises if the goal was dropped
        self._goals.update(updated)
        update = self._updates.add(GoalUpdate(goal_id=goal_id, author_id=actor.id, progress=progress, note=note))
        log.info(
            "Goal %s: progress %s -> %s (%s) by staff %s",
            goal_id,
            goal.progress,
            progress,
            updated.status,
            actor.id,
        )
        return updated, update

    def list_goals(self, actor: StaffMember, trainee_id: int) -> list[Goal]:
        visible_trainee(actor, self._trainees, self._groups, trainee_id)
        return self._goals.list_by_trainee(trainee_id)

    def history(self, actor: StaffMember, goal_id: int) -> list[GoalUpdate]:
        """Every progress update of one goal, oldest first."""
        goal = found(self._goals.get(goal_id), "Goal", goal_id)
        visible_trainee(actor, self._trainees, self._groups, goal.trainee_id, what="Goal", object_id=goal_id)
        return self._updates.list_by_goal(goal_id)

    def _open_trainee(
        self, actor: StaffMember, trainee_id: int, *, what: str = "Trainee", object_id: int | None = None
    ) -> Trainee:
        """The trainee, if this actor may see them (else 404) and their file is still open (else 409).
        Visibility first: "the file is closed" would tell a stranger the trainee's status."""
        trainee = visible_trainee(actor, self._trainees, self._groups, trainee_id, what=what, object_id=object_id)
        if trainee.is_closed:
            raise RuleViolationError(f"Trainee {trainee_id} is {trainee.status}; goals can no longer change")
        return trainee
