"""Endpoints for tasks: every staff member sees and completes their own; the manager sees all."""

from fastapi import APIRouter

from pacer.domain import Task

from ..dependencies import Actor, TaskServiceDep

router = APIRouter(prefix="/tasks", tags=["tasks"])


@router.get("/mine", summary="My tasks")
def my_tasks(actor: Actor, service: TaskServiceDep) -> list[Task]:
    return service.my_tasks(actor)


@router.get("", summary="Everyone's tasks (manager)")
def all_tasks(actor: Actor, service: TaskServiceDep) -> list[Task]:
    return service.all_tasks(actor)


@router.post("/{task_id}/complete", summary="Mark a task done (its owner, or the manager)")
def complete_task(task_id: int, actor: Actor, service: TaskServiceDep) -> Task:
    return service.complete(actor, task_id)
