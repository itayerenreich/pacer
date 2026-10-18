"""Endpoints under /trainees: the trainee, their personal file, status, and goals."""

from fastapi import APIRouter, status

from pacer.domain import Goal, PersonalFile, Trainee, TraineeStatus

from ..dependencies import Actor, GoalServiceDep, TraineeServiceDep
from ..schemas import GoalCreate, StatusChange, TraineeCreate

router = APIRouter(prefix="/trainees", tags=["trainees"])


@router.post("", status_code=status.HTTP_201_CREATED, summary="Enroll a trainee (manager)")
def enroll_trainee(body: TraineeCreate, actor: Actor, service: TraineeServiceDep) -> Trainee:
    return service.enroll(actor, Trainee(**body.model_dump()))


@router.get(
    "",
    summary="List the trainees you may see",
    description="Manager: every trainee. Counselor: the trainees of the groups you lead. "
    "Optionally only one status, e.g. ?status=active.",
)
def list_trainees(actor: Actor, service: TraineeServiceDep, status: TraineeStatus | None = None) -> list[Trainee]:
    return service.list(actor, status)


@router.get("/{trainee_id}", summary="Get a trainee")
def get_trainee(trainee_id: int, actor: Actor, service: TraineeServiceDep) -> Trainee:
    return service.get(actor, trainee_id)


@router.get(
    "/{trainee_id}/personal-file",
    summary="Get the sensitive personal file",
    description="Manager: always. Counselor: only with an approved view_personal_file request.",
)
def get_personal_file(trainee_id: int, actor: Actor, service: TraineeServiceDep) -> PersonalFile:
    return service.get_personal_file(actor, trainee_id)


@router.put(
    "/{trainee_id}/status",
    summary="Change a trainee's status (manager)",
    description="Counselors ask through POST /approvals with kind=status_change.",
)
def change_status(trainee_id: int, body: StatusChange, actor: Actor, service: TraineeServiceDep) -> Trainee:
    return service.change_status(actor, trainee_id, body.status)


@router.get("/{trainee_id}/goals", summary="List a trainee's goals")
def list_goals(trainee_id: int, actor: Actor, service: GoalServiceDep) -> list[Goal]:
    return service.list_goals(actor, trainee_id)


@router.post(
    "/{trainee_id}/goals",
    status_code=status.HTTP_201_CREATED,
    summary="Add a goal (manager or the trainee's primary counselor)",
)
def add_goal(trainee_id: int, body: GoalCreate, actor: Actor, service: GoalServiceDep) -> Goal:
    return service.add_goal(actor, Goal(trainee_id=trainee_id, **body.model_dump()))
