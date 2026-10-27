"""Tasks written by hand (2026-10-27): who writes for whom, the trainee a task is about, deadlines,
deleting, and the order a staff member sees them in."""

from datetime import date, timedelta

import pytest
from fakes import World

from pacer.domain import DomainRuleError, Task, TaskStatus
from pacer.services import NotFoundError, PermissionDeniedError, RuleViolationError, TaskService

TODAY = date(2026, 10, 27)


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def service(world) -> TaskService:
    return TaskService(world.tasks, world.staff, world.trainees, world.groups)


def write(service, actor, text="לדבר עם החניכה על הריצה", **kwargs) -> Task:
    return service.create(actor, text, today=TODAY, **kwargs)


# --- who writes for whom -------------------------------------------------------------------------


def test_a_counselor_writes_a_task_for_themselves(world, service):
    task = write(service, world.noa, due_date=TODAY + timedelta(days=2))
    assert (task.owner_id, task.created_by_id, task.status, task.written_by_hand) == (
        world.noa.id,
        world.noa.id,
        TaskStatus.OPEN,
        True,
    )


def test_the_manager_writes_a_task_for_anyone(world, service):
    task = write(service, world.manager, owner_id=world.dana.id)
    assert (task.owner_id, task.created_by_id) == (world.dana.id, world.manager.id)


def test_a_counselor_cannot_write_for_someone_else_and_learns_nothing(world, service):
    with pytest.raises(PermissionDeniedError):
        write(service, world.noa, owner_id=world.dana.id)
    with pytest.raises(PermissionDeniedError):  # the same answer for an id that does not exist: no probing
        write(service, world.noa, owner_id=999)


def test_the_owner_must_exist(world, service):
    with pytest.raises(NotFoundError):
        write(service, world.manager, owner_id=999)


# --- the trainee a task is about -----------------------------------------------------------------


def test_a_task_about_a_trainee_shows_in_their_profile(world, service):
    task = write(service, world.noa, trainee_id=world.yael.id)
    assert service.for_trainee(world.noa, world.yael.id) == [task]


def test_a_task_about_a_trainee_the_writer_may_not_see_is_404_like_one_that_does_not_exist(world, service):
    with pytest.raises(NotFoundError) as hidden:
        write(service, world.dana, trainee_id=world.yael.id)
    with pytest.raises(NotFoundError) as missing:
        write(service, world.dana, trainee_id=999)
    assert str(hidden.value).replace(str(world.yael.id), "N") == str(missing.value).replace("999", "N")


def test_the_manager_cannot_give_a_task_about_a_trainee_to_someone_who_cannot_open_them(world, service):
    with pytest.raises(RuleViolationError, match="does not work with trainee"):
        write(service, world.manager, owner_id=world.dana.id, trainee_id=world.yael.id)


def test_in_the_profile_a_counselor_sees_their_own_tasks_and_the_manager_everyones(world, service):
    mine = write(service, world.noa, trainee_id=world.yael.id)
    omers = write(service, world.omer, trainee_id=world.yael.id)
    assert service.for_trainee(world.noa, world.yael.id) == [mine]
    assert {t.key for t in service.for_trainee(world.manager, world.yael.id)} == {mine.key, omers.key}
    with pytest.raises(NotFoundError):
        service.for_trainee(world.dana, world.yael.id)


# --- deadlines -----------------------------------------------------------------------------------


def test_a_deadline_is_never_in_the_past_today_is_fine(world, service):
    with pytest.raises(RuleViolationError, match="already passed"):
        write(service, world.noa, due_date=TODAY - timedelta(days=1))
    assert write(service, world.noa, due_date=TODAY).due_date == TODAY


def test_the_deadline_changes_or_goes_away_while_the_task_is_open(world, service):
    task = write(service, world.noa, due_date=TODAY)
    assert service.reschedule(world.noa, task.key, TODAY + timedelta(days=7), today=TODAY).due_date == TODAY + timedelta(days=7)
    assert service.reschedule(world.noa, task.key, None, today=TODAY).due_date is None
    service.complete(world.noa, task.key)
    with pytest.raises(DomainRuleError):
        service.reschedule(world.noa, task.key, TODAY, today=TODAY)


def test_someone_elses_task_is_404_for_a_counselor_and_open_to_the_manager(world, service):
    task = write(service, world.noa)
    for action in (
        lambda: service.complete(world.dana, task.key),
        lambda: service.reschedule(world.dana, task.key, None, today=TODAY),
        lambda: service.delete(world.dana, task.key),
    ):
        with pytest.raises(NotFoundError):
            action()
    assert service.reschedule(world.manager, task.key, TODAY, today=TODAY).due_date == TODAY


# --- deleting ------------------------------------------------------------------------------------


def test_a_task_written_by_hand_is_deleted_by_whoever_wrote_it(world, service):
    task = write(service, world.noa)
    service.delete(world.noa, task.key)
    assert world.tasks.get(task.key) is None


def test_a_task_the_manager_gave_is_not_deleted_by_its_owner(world, service):
    task = write(service, world.manager, owner_id=world.noa.id)
    with pytest.raises(PermissionDeniedError):
        service.delete(world.noa, task.key)
    service.delete(world.manager, task.key)


def test_a_task_from_a_staff_meeting_is_never_deleted_only_done(world, service):
    task = world.tasks.add(Task(description="להכין את הטיול", owner_id=world.noa.id, created_by_id=world.manager.id, source_meeting_id=1))
    with pytest.raises(RuleViolationError, match="mark it done"):
        service.delete(world.manager, task.key)


# --- the order -----------------------------------------------------------------------------------


def test_open_first_the_nearest_deadline_first_no_deadline_last_then_done(world, service):
    later = write(service, world.noa, "משימה בעוד שבוע", due_date=TODAY + timedelta(days=7))
    none = write(service, world.noa, "משימה בלי תאריך")
    soon = write(service, world.noa, "משימה למחר", due_date=TODAY + timedelta(days=1))
    done = write(service, world.noa, "משימה שבוצעה", due_date=TODAY)
    service.complete(world.noa, done.key)
    assert [t.key for t in service.my_tasks(world.noa)] == [soon.key, later.key, none.key, done.key]
