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
"""

from collections.abc import Iterator
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from pacer.domain import StaffMember
from pacer.repositories import (
    ApprovalRequestRepository,
    GoalRepository,
    GoalUpdateRepository,
    GroupRepository,
    PersonalFileRepository,
    StaffRepository,
    TraineeRepository,
    get_engine,
    unit_of_work,
)
from pacer.repositories.interfaces import (
    ApprovalRequestRepo,
    GoalRepo,
    GoalUpdateRepo,
    GroupRepo,
    PersonalFileRepo,
    StaffRepo,
    TraineeRepo,
)
from pacer.services import ApprovalService, GoalService, TraineeService

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


# --- who is asking ----------------------------------------------------------------------


def get_current_actor(
    staff: Annotated[StaffRepo, Depends(get_staff_repository)],
    x_staff_id: Annotated[
        int | None, Header(description="Development only: the id of the acting staff member")
    ] = None,
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
