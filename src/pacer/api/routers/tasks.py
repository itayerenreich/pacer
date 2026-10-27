"""Endpoints for tasks: every staff member writes, sees and completes their own; the manager also writes
tasks for anyone and sees everyone's. A task can be about a trainee (decision 2026-10-27)."""

from fastapi import APIRouter, Response, status

from pacer.domain import Task

from ..dependencies import Actor, TaskServiceDep, TodayDep
from ..schemas import TaskChange, TaskCreate

router = APIRouter(tags=["tasks"])


@router.get("/tasks/mine", summary="My tasks: open first, the nearest deadline first")
def my_tasks(actor: Actor, service: TaskServiceDep) -> list[Task]:
    return service.my_tasks(actor)


@router.get("/tasks", summary="Everyone's tasks (manager)")
def all_tasks(actor: Actor, service: TaskServiceDep) -> list[Task]:
    return service.all_tasks(actor)


@router.post(
    "/tasks",
    status_code=status.HTTP_201_CREATED,
    summary="Write a task: for yourself, or (the manager) for anyone. Optionally about a trainee",
)
def create_task(body: TaskCreate, actor: Actor, service: TaskServiceDep, today: TodayDep) -> Task:
    return service.create(
        actor,
        body.description,
        today=today,
        due_date=body.due_date,
        owner_id=body.owner_id,
        trainee_id=body.trainee_id,
    )


@router.post("/tasks/{task_id}/complete", summary="Mark a task done (its owner, or the manager)")
def complete_task(task_id: int, actor: Actor, service: TaskServiceDep) -> Task:
    return service.complete(actor, task_id)


@router.patch("/tasks/{task_id}", summary="Change an open task's deadline (its owner, or the manager)")
def change_task(task_id: int, body: TaskChange, actor: Actor, service: TaskServiceDep, today: TodayDep) -> Task:
    return service.reschedule(actor, task_id, body.due_date, today=today)


@router.delete(
    "/tasks/{task_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a task written by hand (whoever wrote it, or the manager). A meeting's task: mark it done",
)
def delete_task(task_id: int, actor: Actor, service: TaskServiceDep) -> Response:
    service.delete(actor, task_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/trainees/{trainee_id}/tasks", summary="Tasks about a trainee: yours (the manager: everyone's)")
def trainee_tasks(trainee_id: int, actor: Actor, service: TaskServiceDep) -> list[Task]:
    return service.for_trainee(actor, trainee_id)
