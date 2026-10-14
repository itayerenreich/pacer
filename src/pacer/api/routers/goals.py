"""Endpoints under /goals: progress updates and their history."""

from fastapi import APIRouter

from pacer.domain import Goal, GoalUpdate

from ..dependencies import Actor, GoalServiceDep
from ..schemas import ProgressUpdate

router = APIRouter(prefix="/goals", tags=["goals"])


@router.post(
    "/{goal_id}/progress",
    summary="Update a goal's progress",
    description=(
        "Saves the new progress on the goal AND a history entry (who, when, why), together. "
        "The status follows the progress: 0 not started, 1-99 in progress, 100 achieved."
    ),
)
def update_progress(goal_id: int, body: ProgressUpdate, actor: Actor, service: GoalServiceDep) -> Goal:
    return service.update_progress(actor, goal_id, body.progress, body.note)


@router.get("/{goal_id}/history", summary="A goal's progress history, oldest first")
def goal_history(goal_id: int, actor: Actor, service: GoalServiceDep) -> list[GoalUpdate]:
    return service.history(actor, goal_id)
