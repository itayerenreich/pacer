"""Core domain models (Trainee, Goal, StaffMember, ApprovalRequest...).

Pure data and rules. Must not import from any other layer.
"""

from .approval import ApprovalRequest
from .cohort import Cohort, Group
from .entity import Entity, present
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
from .library import CHUNK_MAX_LENGTH, Chunk, EvidenceLevel, KnowledgeSource, SourceKind, SourceLicense
from .meeting import ProposedTask, StaffMeeting, Task
from .schedule import (
    ACADEMY_TIMEZONE,
    AttendanceRecord,
    AttendanceStatus,
    ScheduledSession,
    ScheduleSlot,
    Weekday,
    academy_today,
    week_start,
)
from .staff import StaffMember
from .talk import PersonalTalk, TalkProposal, TraineeNote
from .trainee import PersonalFile, Trainee

__all__ = [
    "ACADEMY_TIMEZONE",
    "academy_today",
    "AttendanceRecord",
    "AttendanceStatus",
    "ScheduleSlot",
    "ScheduledSession",
    "Weekday",
    "week_start",
    "CHUNK_MAX_LENGTH",
    "Chunk",
    "EvidenceLevel",
    "KnowledgeSource",
    "SourceKind",
    "SourceLicense",
    "Entity",
    "present",
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
