"""Core domain models (Trainee, Goal, Counselor).

Pure data and rules. Must not import from any other layer.
"""

from .cohort import Cohort, Group
from .enums import (
    GoalCategory,
    GoalStatus,
    ResponsibilityArea,
    StaffRole,
    TraineeStatus,
)
from .goal import Goal, GoalUpdate
from .staff import StaffMember
from .trainee import PersonalFile, Trainee

__all__ = [
    "Cohort",
    "Goal",
    "GoalCategory",
    "GoalStatus",
    "GoalUpdate",
    "Group",
    "PersonalFile",
    "ResponsibilityArea",
    "StaffMember",
    "StaffRole",
    "Trainee",
    "TraineeStatus",
]
