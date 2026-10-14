"""Business logic: goal progress, permissions, approvals.

May import from: domain, repositories.

A service method always gets the acting staff member (the "actor") first, checks
who may do this, checks the rules that involve more than one object, and only then
reads or writes through the repositories it was given.

Usage (the API will do this per request):
    with unit_of_work() as session:
        goals = GoalService(
            GoalRepository(session), GoalUpdateRepository(session),
            TraineeRepository(session), GroupRepository(session),
        )
        goals.update_progress(actor, goal_id=7, progress=60, note="Ran 3 km")
"""

from .approvals import ApprovalService
from .errors import NotFoundError, PermissionDeniedError, RuleViolationError, ServiceError
from .goals import GoalService
from .trainees import TraineeService

__all__ = [
    "ApprovalService",
    "GoalService",
    "NotFoundError",
    "PermissionDeniedError",
    "RuleViolationError",
    "ServiceError",
    "TraineeService",
]
