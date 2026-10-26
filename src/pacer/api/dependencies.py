"""Dependency Injection: how every request gets its database session, actor and services.

FastAPI builds this chain for each request, from the bottom up:

    get_engine ──> get_session ──> get_*_repository ──> get_*_service ──> endpoint
                                        └──> get_current_actor ─────────────┘

* One request = one unit of work. FastAPI caches a dependency within a request,
  so every repository and service in the same request shares ONE session, and
  all their writes are committed together (or rolled back together).
* scope="function": the unit of work commits BEFORE the response is sent. With
  FastAPI's default ("request"), the commit would run after the client already got
  "200 OK", and a failed commit would be invisible to the client.
* Tests swap any link in the chain with app.dependency_overrides, e.g. a test
  database engine, or in-memory fake repositories with no database at all.

Endpoints that wait for a model (the AI pipelines) are different: they must NOT hold a
transaction open while the model works. They never touch SessionDep. They get a unit-of-work
FACTORY instead, and the pipeline opens short transactions of its own (read -> model -> write).
Even the actor is read in its own short transaction (DetachedActor), closed before the endpoint runs.
"""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from pacer.ai.answer import LibraryAnswerer
from pacer.ai.coach import Coach
from pacer.ai.embeddings import DEFAULT_EMBEDDING_MODEL, Embedder, ollama_embedder
from pacer.ai.meeting_pipeline import MeetingPipeline
from pacer.ai.routing import Router, get_router
from pacer.ai.talk_pipeline import TalkPipeline
from pacer.domain import StaffMember
from pacer.library.chunking import DEFAULT_CHUNKING
from pacer.repositories import (
    ApprovalRequestRepository,
    GoalRepository,
    GoalUpdateRepository,
    GroupRepository,
    MeetingRepository,
    PersonalFileRepository,
    PersonalTalkRepository,
    ProposedTaskRepository,
    StaffRepository,
    TalkProposalRepository,
    TaskRepository,
    TraineeNoteRepository,
    TraineeRepository,
    UnitOfWork,
    get_engine,
    repositories,
    unit_of_work,
)
from pacer.repositories.interfaces import (
    ApprovalRequestRepo,
    GoalRepo,
    GoalUpdateRepo,
    GroupRepo,
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
from pacer.services import (
    ApprovalService,
    GoalService,
    MeetingService,
    TalkService,
    TaskService,
    TraineeService,
)

from .errors import NotAuthenticatedError

# --- database -------------------------------------------------------------------------


def get_db_engine() -> Engine:
    """The production engine (from .env). Tests override this with a test database."""
    return get_engine()


def get_session(engine: Annotated[Engine, Depends(get_db_engine)]) -> Iterator[Session]:
    """One unit of work per request: commit if the endpoint succeeds, roll back if it raises."""
    with unit_of_work(engine) as session:
        yield session


SessionDep = Annotated[Session, Depends(get_session, scope="function")]


# --- repositories ---------------------------------------------------------------------


def get_staff_repository(session: SessionDep) -> StaffRepo:
    return StaffRepository(session)


def get_group_repository(session: SessionDep) -> GroupRepo:
    return GroupRepository(session)


def get_trainee_repository(session: SessionDep) -> TraineeRepo:
    return TraineeRepository(session)


def get_personal_file_repository(session: SessionDep) -> PersonalFileRepo:
    return PersonalFileRepository(session)


def get_goal_repository(session: SessionDep) -> GoalRepo:
    return GoalRepository(session)


def get_goal_update_repository(session: SessionDep) -> GoalUpdateRepo:
    return GoalUpdateRepository(session)


def get_approval_repository(session: SessionDep) -> ApprovalRequestRepo:
    return ApprovalRequestRepository(session)


def get_meeting_repository(session: SessionDep) -> MeetingRepo:
    return MeetingRepository(session)


def get_proposal_repository(session: SessionDep) -> ProposedTaskRepo:
    return ProposedTaskRepository(session)


def get_task_repository(session: SessionDep) -> TaskRepo:
    return TaskRepository(session)


def get_talk_repository(session: SessionDep) -> PersonalTalkRepo:
    return PersonalTalkRepository(session)


def get_talk_proposal_repository(session: SessionDep) -> TalkProposalRepo:
    return TalkProposalRepository(session)


def get_note_repository(session: SessionDep) -> TraineeNoteRepo:
    return TraineeNoteRepository(session)


# --- who is asking ----------------------------------------------------------------------


StaffIdHeader = Annotated[int | None, Header(description="Development only: the id of the acting staff member")]


def get_current_actor(
    staff: Annotated[StaffRepo, Depends(get_staff_repository)], x_staff_id: StaffIdHeader = None
) -> StaffMember:
    """The staff member making this request.

    DEVELOPMENT ONLY: the client says who it is in the X-Staff-Id header, and we
    believe it. Anyone can claim to be the manager. This stands in for real login
    (e.g. OAuth2 + JWT) so the permission rules can be built and tested now. Only
    this function changes when real authentication arrives; endpoints and services
    keep receiving a StaffMember.
    """
    if x_staff_id is None:
        raise NotAuthenticatedError("Missing X-Staff-Id header")
    actor = staff.get(x_staff_id)
    if actor is None:
        raise NotAuthenticatedError(f"No staff member with id {x_staff_id}")
    return actor


Actor = Annotated[StaffMember, Depends(get_current_actor)]


def get_unit_of_work(engine: Annotated[Engine, Depends(get_db_engine)]) -> UnitOfWork:
    """A factory of short transactions, for code that must open and close its own."""
    return lambda: repositories(engine)


UnitOfWorkDep = Annotated[UnitOfWork, Depends(get_unit_of_work)]


def get_detached_actor(uow: UnitOfWorkDep, x_staff_id: StaffIdHeader = None) -> StaffMember:
    """The same check as get_current_actor, in a transaction that is closed again before the
    endpoint runs. For endpoints that wait for a model."""
    with uow() as repos:
        return get_current_actor(repos.staff, x_staff_id)


DetachedActor = Annotated[StaffMember, Depends(get_detached_actor)]


# --- services -------------------------------------------------------------------------


def get_goal_service(
    goals: Annotated[GoalRepo, Depends(get_goal_repository)],
    updates: Annotated[GoalUpdateRepo, Depends(get_goal_update_repository)],
    trainees: Annotated[TraineeRepo, Depends(get_trainee_repository)],
    groups: Annotated[GroupRepo, Depends(get_group_repository)],
) -> GoalService:
    return GoalService(goals, updates, trainees, groups)


def get_trainee_service(
    trainees: Annotated[TraineeRepo, Depends(get_trainee_repository)],
    groups: Annotated[GroupRepo, Depends(get_group_repository)],
    personal_files: Annotated[PersonalFileRepo, Depends(get_personal_file_repository)],
    approvals: Annotated[ApprovalRequestRepo, Depends(get_approval_repository)],
) -> TraineeService:
    return TraineeService(trainees, groups, personal_files, approvals)


def get_approval_service(
    approvals: Annotated[ApprovalRequestRepo, Depends(get_approval_repository)],
    trainees: Annotated[TraineeRepo, Depends(get_trainee_repository)],
    groups: Annotated[GroupRepo, Depends(get_group_repository)],
) -> ApprovalService:
    return ApprovalService(approvals, trainees, groups)


GoalServiceDep = Annotated[GoalService, Depends(get_goal_service)]
TraineeServiceDep = Annotated[TraineeService, Depends(get_trainee_service)]
ApprovalServiceDep = Annotated[ApprovalService, Depends(get_approval_service)]


def get_meeting_service(
    meetings: Annotated[MeetingRepo, Depends(get_meeting_repository)],
    proposals: Annotated[ProposedTaskRepo, Depends(get_proposal_repository)],
    tasks: Annotated[TaskRepo, Depends(get_task_repository)],
    staff: Annotated[StaffRepo, Depends(get_staff_repository)],
) -> MeetingService:
    return MeetingService(meetings, proposals, tasks, staff)


def get_task_service(tasks: Annotated[TaskRepo, Depends(get_task_repository)]) -> TaskService:
    return TaskService(tasks)


MeetingServiceDep = Annotated[MeetingService, Depends(get_meeting_service)]
TaskServiceDep = Annotated[TaskService, Depends(get_task_service)]


def get_talk_service(
    talks: Annotated[PersonalTalkRepo, Depends(get_talk_repository)],
    proposals: Annotated[TalkProposalRepo, Depends(get_talk_proposal_repository)],
    notes: Annotated[TraineeNoteRepo, Depends(get_note_repository)],
    trainees: Annotated[TraineeRepo, Depends(get_trainee_repository)],
    groups: Annotated[GroupRepo, Depends(get_group_repository)],
    goals: Annotated[GoalRepo, Depends(get_goal_repository)],
    goal_service: GoalServiceDep,
) -> TalkService:
    return TalkService(talks, proposals, notes, trainees, groups, goals, goal_service)


TalkServiceDep = Annotated[TalkService, Depends(get_talk_service)]


# --- AI ---------------------------------------------------------------------------------------


def get_ai_router() -> Router:
    return get_router()


def get_meeting_pipeline(
    uow: UnitOfWorkDep,
    router: Annotated[Router, Depends(get_ai_router)],
) -> MeetingPipeline:
    return MeetingPipeline(uow, router)


MeetingPipelineDep = Annotated[MeetingPipeline, Depends(get_meeting_pipeline)]


def get_talk_pipeline(
    uow: UnitOfWorkDep, router: Annotated[Router, Depends(get_ai_router)]
) -> TalkPipeline:
    return TalkPipeline(uow, router)


TalkPipelineDep = Annotated[TalkPipeline, Depends(get_talk_pipeline)]


def get_embedder() -> Embedder:
    """The embedding model of the library (chosen by measurement), served by Ollama on this computer."""
    return ollama_embedder(DEFAULT_EMBEDDING_MODEL)


def get_library_answerer(
    uow: UnitOfWorkDep,
    router: Annotated[Router, Depends(get_ai_router)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> LibraryAnswerer:
    return LibraryAnswerer(uow, router, embedder, DEFAULT_CHUNKING)


LibraryAnswererDep = Annotated[LibraryAnswerer, Depends(get_library_answerer)]


def get_coach(
    uow: UnitOfWorkDep,
    router: Annotated[Router, Depends(get_ai_router)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> Coach:
    return Coach(uow, router, embedder, DEFAULT_CHUNKING)


CoachDep = Annotated[Coach, Depends(get_coach)]
