"""Persistence layer: saves and loads domain objects from the database.

May import from: domain.
"""

from .approval import ApprovalRequestRepository
from .cohort import CohortRepository, GroupRepository
from .database import get_engine, unit_of_work
from .goal import GoalRepository, GoalUpdateRepository
from .interfaces import ScoredChunk
from .library import ChunkRepository, KnowledgeSourceRepository
from .meeting import MeetingRepository, ProposedTaskRepository, TaskRepository
from .repos import Repos, UnitOfWork, repositories
from .staff import StaffRepository
from .talk import PersonalTalkRepository, TalkProposalRepository, TraineeNoteRepository
from .trainee import PersonalFileRepository, TraineeRepository

__all__ = [
    "ChunkRepository",
    "KnowledgeSourceRepository",
    "ScoredChunk",
    "ApprovalRequestRepository",
    "CohortRepository",
    "GoalRepository",
    "GoalUpdateRepository",
    "GroupRepository",
    "MeetingRepository",
    "PersonalFileRepository",
    "PersonalTalkRepository",
    "ProposedTaskRepository",
    "Repos",
    "StaffRepository",
    "TalkProposalRepository",
    "TaskRepository",
    "TraineeNoteRepository",
    "TraineeRepository",
    "UnitOfWork",
    "get_engine",
    "repositories",
    "unit_of_work",
]
