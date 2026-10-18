"""Every repository over one session: a unit of work that hands out repositories.

    with repositories(engine) as repos:
        meeting = repos.meetings.get(7)
        repos.proposals.add(...)
    # committed here (or rolled back, if the block raised)

Services get the few repositories they need in their constructor. Code that has to open and
close its OWN transactions, like an AI pipeline that must not hold one open while a model
thinks, gets a factory instead: a function that returns a fresh `repositories(...)` each time.
Tests pass a factory that yields in-memory fakes, so the same code runs without a database.
"""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from .approval import ApprovalRequestRepository
from .cohort import CohortRepository, GroupRepository
from .database import unit_of_work
from .goal import GoalRepository, GoalUpdateRepository
from .interfaces import (
    ApprovalRequestRepo,
    ChunkRepo,
    CohortRepo,
    GoalRepo,
    GoalUpdateRepo,
    GroupRepo,
    KnowledgeSourceRepo,
    MeetingRepo,
    PersonalFileRepo,
    PersonalTalkRepo,
    ProposedTaskRepo,
    StaffRepo,
    TalkProposalRepo,
    TaskRepo,
    TraineeNoteRepo,
    TraineeRepo,
)
from .library import ChunkRepository, KnowledgeSourceRepository
from .meeting import MeetingRepository, ProposedTaskRepository, TaskRepository
from .staff import StaffRepository
from .talk import PersonalTalkRepository, TalkProposalRepository, TraineeNoteRepository
from .trainee import PersonalFileRepository, TraineeRepository


@dataclass(frozen=True)
class Repos:
    """Every repository, typed by its Protocol: SQLAlchemy ones over one session, or fakes."""

    staff: StaffRepo
    cohorts: CohortRepo
    groups: GroupRepo
    trainees: TraineeRepo
    personal_files: PersonalFileRepo
    goals: GoalRepo
    goal_updates: GoalUpdateRepo
    approvals: ApprovalRequestRepo
    meetings: MeetingRepo
    proposals: ProposedTaskRepo
    tasks: TaskRepo
    talks: PersonalTalkRepo
    talk_proposals: TalkProposalRepo
    notes: TraineeNoteRepo
    sources: KnowledgeSourceRepo
    chunks: ChunkRepo

    @classmethod
    def over(cls, session: Session) -> "Repos":
        return cls(
            staff=StaffRepository(session),
            cohorts=CohortRepository(session),
            groups=GroupRepository(session),
            trainees=TraineeRepository(session),
            personal_files=PersonalFileRepository(session),
            goals=GoalRepository(session),
            goal_updates=GoalUpdateRepository(session),
            approvals=ApprovalRequestRepository(session),
            meetings=MeetingRepository(session),
            proposals=ProposedTaskRepository(session),
            tasks=TaskRepository(session),
            talks=PersonalTalkRepository(session),
            talk_proposals=TalkProposalRepository(session),
            notes=TraineeNoteRepository(session),
            sources=KnowledgeSourceRepository(session),
            chunks=ChunkRepository(session),
        )


UnitOfWork = Callable[[], AbstractContextManager[Repos]]
"""A factory of transactions: each call opens a new one. `with uow() as repos: ...`"""


@contextmanager
def repositories(engine: Engine | None = None) -> Iterator[Repos]:
    with unit_of_work(engine) as session:
        yield Repos.over(session)
