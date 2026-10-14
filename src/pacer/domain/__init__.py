"""Core domain models (Trainee, Goal, StaffMember, ApprovalRequest...).

Pure data and rules. Must not import from any other layer.
"""

from .approval import ApprovalRequest
from .cohort import Cohort, Group
from .enums import (
    ApprovalKind,
    ApprovalStatus,
    GoalCategory,
    GoalStatus,
    MeetingStatus,
    ProposalStatus,
    ResponsibilityArea,
    StaffRole,
    TaskStatus,
    TraineeStatus,
)
from .errors import DomainRuleError, InvalidTransitionError
from .goal import Goal, GoalUpdate
from .meeting import ProposedTask, StaffMeeting, Task
from .staff import StaffMember
from .trainee import PersonalFile, Trainee

__all__ = [
    "ApprovalKind",
    "ApprovalRequest",
    "ApprovalStatus",
    "Cohort",
    "DomainRuleError",
    "Goal",
    "GoalCategory",
    "GoalStatus",
    "GoalUpdate",
    "Group",
    "InvalidTransitionError",
    "MeetingStatus",
    "ProposalStatus",
    "ProposedTask",
    "PersonalFile",
    "ResponsibilityArea",
    "StaffMeeting",
    "StaffMember",
    "StaffRole",
    "Task",
    "TaskStatus",
    "Trainee",
    "TraineeStatus",
]
