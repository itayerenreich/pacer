"""Tasks: from a staff meeting, or written by hand (2026-10-27), each with an optional deadline and
an optional trainee it is about.

Who may do what (decision 2026-10-27):
    Staff member  sees, writes and completes their OWN tasks, and changes their deadline.
    Manager       all of that for anyone: writes a task for any staff member, sees everyone's.
    Deleting      only a task written by hand, by whoever wrote it (or the manager). A task from a
                  meeting was decided by the manager, and that decision stays on record.

A task about a trainee: the writer must see the trainee (404 otherwise), and so must the owner (409:
a task about someone you cannot open is a task you cannot do). In a trainee's profile, each staff
member sees their own tasks about them; the manager sees everyone's.

"Today" comes from the caller (the academy's day, Asia/Jerusalem): a new deadline is never in the past.
"""

import logging
from datetime import date

from pacer.domain import StaffMember, Task, TaskStatus
from pacer.repositories.interfaces import GroupRepo, StaffRepo, TaskRepo, TraineeRepo

from .errors import RuleViolationError, found
from .permissions import can_view_trainee, is_manager, require
from .visibility import hidden, visible_trainee

log = logging.getLogger(__name__)


class TaskService:
    def __init__(self, tasks: TaskRepo, staff: StaffRepo, trainees: TraineeRepo, groups: GroupRepo) -> None:
        self._tasks = tasks
        self._staff = staff
        self._trainees = trainees
        self._groups = groups

    # --- reading ---------------------------------------------------------------------------------

    def my_tasks(self, actor: StaffMember) -> list[Task]:
        return _by_deadline(self._tasks.list_by_owner(actor.key))

    def all_tasks(self, actor: StaffMember) -> list[Task]:
        require(is_manager(actor), "view everyone's tasks")
        return _by_deadline(self._tasks.list_all())

    def for_trainee(self, actor: StaffMember, trainee_id: int) -> list[Task]:
        """The tasks about a trainee this actor may see: their own, or everyone's for the manager."""
        visible_trainee(actor, self._trainees, self._groups, trainee_id)
        tasks = self._tasks.list_by_trainee(trainee_id)
        return _by_deadline(tasks if is_manager(actor) else [t for t in tasks if t.owner_id == actor.key])

    # --- writing ---------------------------------------------------------------------------------

    def create(
        self,
        actor: StaffMember,
        description: str,
        *,
        today: date,
        due_date: date | None = None,
        owner_id: int | None = None,
        trainee_id: int | None = None,
    ) -> Task:
        owner_id = actor.key if owner_id is None else owner_id
        if owner_id != actor.key:
            require(is_manager(actor), "write a task for someone else")  # 403 before any lookup
        owner = found(self._staff.get(owner_id), "Staff member", owner_id)
        _not_past(due_date, today)
        if trainee_id is not None:
            trainee = visible_trainee(actor, self._trainees, self._groups, trainee_id)
            group = found(self._groups.get(trainee.group_id), "Group", trainee.group_id)
            if not can_view_trainee(owner, group):
                raise RuleViolationError(
                    f"Staff member {owner_id} does not work with trainee {trainee_id}: they could not open the task"
                )
        task = self._tasks.add(
            Task(
                description=description,
                owner_id=owner.key,
                due_date=due_date,
                created_by_id=actor.key,
                trainee_id=trainee_id,
            )
        )
        log.info("Task %s written by staff %s for staff %s (trainee %s)", task.id, actor.id, owner_id, trainee_id)
        return task

    def complete(self, actor: StaffMember, task_id: int) -> Task:
        task = self._visible(actor, task_id)
        done = self._tasks.update(task.complete())  # InvalidTransition if already done
        log.info("Task %s completed by staff %s", task_id, actor.id)
        return done

    def reschedule(self, actor: StaffMember, task_id: int, due_date: date | None, *, today: date) -> Task:
        task = self._visible(actor, task_id)
        _not_past(due_date, today)
        changed = self._tasks.update(task.with_due_date(due_date))  # a done task: DomainRuleError (409)
        log.info("Task %s: new deadline, by staff %s", task_id, actor.id)
        return changed

    def delete(self, actor: StaffMember, task_id: int) -> None:
        task = self._visible(actor, task_id)
        if not task.written_by_hand:
            raise RuleViolationError(f"Task {task_id} came from a staff meeting: mark it done instead of deleting it")
        if task.created_by_id != actor.key and not is_manager(actor):
            require(False, "delete a task someone else wrote")
        self._tasks.delete(task_id)
        log.info("Task %s deleted by staff %s", task_id, actor.id)

    def _visible(self, actor: StaffMember, task_id: int) -> Task:
        """A staff member sees only their own tasks: someone else's is a 404, as if it did not exist."""
        task = found(self._tasks.get(task_id), "Task", task_id)
        if task.owner_id != actor.key and not is_manager(actor):
            raise hidden("Task", task_id)
        return task


def _not_past(due_date: date | None, today: date) -> None:
    if due_date is not None and due_date < today:
        raise RuleViolationError(f"The deadline {due_date} has already passed")


def _by_deadline(tasks: list[Task]) -> list[Task]:
    """Open first; among them the nearest deadline first, and no deadline last. Then the done ones."""
    return sorted(
        tasks, key=lambda t: (t.status is not TaskStatus.OPEN, t.due_date is None, t.due_date or date.max, t.key)
    )
