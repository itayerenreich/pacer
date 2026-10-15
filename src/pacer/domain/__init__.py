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
    NoteKind,
    ProposalStatus,
    ResponsibilityArea,
    StaffRole,
    TalkProposalKind,
    TalkStatus,
    TaskStatus,
    TraineeStatus,
)
from .errors import DomainRuleError, InvalidTransitionError
from .goal import Goal, GoalUpdate
from .meeting import ProposedTask, StaffMeeting, Task
from .staff import StaffMember
from .talk import PersonalTalk, TalkProposal, TraineeNote
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
    "NoteKind",
    "ProposalStatus",
    "ProposedTask",
    "PersonalFile",
    "PersonalTalk",
    "ResponsibilityArea",
    "StaffMeeting",
    "StaffMember",
    "StaffRole",
    "TalkProposal",
    "TalkProposalKind",
    "TalkStatus",
    "Task",
    "TaskStatus",
    "Trainee",
    "TraineeNote",
    "TraineeStatus",
]
