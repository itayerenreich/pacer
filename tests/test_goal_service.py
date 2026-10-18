"""GoalService tests with in-memory fakes (see fakes.py). No database needed."""

import pytest
from fakes import World

from pacer.domain import DomainRuleError, Goal, GoalStatus, TraineeStatus
from pacer.services import GoalService, NotFoundError, PermissionDeniedError, RuleViolationError


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def service(world) -> GoalService:
    return GoalService(world.goals, world.goal_updates, world.trainees, world.groups)


def close_file(world: World, status: TraineeStatus) -> None:
    world.trainees.update(world.yael.with_status(status))


# --- update_progress ---------------------------------------------------------------


def test_update_progress_saves_goal_and_history(world, service):
    updated = service.update_progress(world.noa, world.goal.id, 40, "Ran 2 km without stopping")

    assert updated.progress == 40
    assert updated.status is GoalStatus.IN_PROGRESS
    assert world.goals.get(world.goal.id).progress == 40  # the stored goal changed
    [entry] = world.goal_updates.list_by_goal(world.goal.id)
    assert (entry.author_id, entry.progress, entry.note) == (world.noa.id, 40, "Ran 2 km without stopping")


def test_history_keeps_every_step_in_order(world, service):
    for progress in (20, 50, 100):
        service.update_progress(world.noa, world.goal.id, progress, f"Now at {progress}")

    history = service.history(world.omer, world.goal.id)  # the co-counselor may read it

    assert [u.progress for u in history] == [20, 50, 100]
    assert world.goals.get(world.goal.id).status is GoalStatus.ACHIEVED


def test_manager_can_update_progress(world, service):
    assert service.update_progress(world.manager, world.goal.id, 10, "Started").progress == 10


@pytest.mark.parametrize(
    ("counselor", "error"),
    [("omer", PermissionDeniedError), ("dana", NotFoundError)],  # same group: sees, may not edit. Other: 404
)
def test_other_counselors_cannot_update_progress(world, service, counselor, error):
    with pytest.raises(error):
        service.update_progress(getattr(world, counselor), world.goal.id, 40, "Trying")
    assert world.goal_updates.list_by_goal(world.goal.id) == []


@pytest.mark.parametrize("status", [TraineeStatus.LEFT, TraineeStatus.GRADUATED])
def test_closed_trainee_goals_are_frozen(world, service, status):
    close_file(world, status)
    with pytest.raises(RuleViolationError, match="no longer change"):
        service.update_progress(world.noa, world.goal.id, 40, "Too late")


def test_dropped_goal_cannot_progress(world, service):
    world.goals.update(world.goal.model_copy(update={"status": GoalStatus.DROPPED}))
    with pytest.raises(DomainRuleError):
        service.update_progress(world.noa, world.goal.id, 40, "Back on it")


def test_missing_goal_is_not_found(world, service):
    with pytest.raises(NotFoundError, match="Goal 999"):
        service.update_progress(world.noa, 999, 40, "Hello?")


# --- add_goal / list / history -----------------------------------------------------


def test_primary_counselor_adds_a_goal(world, service):
    goal = service.add_goal(
        world.noa, Goal(trainee_id=world.yael.id, title="Lead a discussion", category="leadership")
    )
    assert [g.title for g in service.list_goals(world.noa, world.yael.id)] == [
        "Run 5 km",
        "Lead a discussion",
    ]
    assert goal.id is not None


def test_no_new_goals_for_a_trainee_who_left(world, service):
    close_file(world, TraineeStatus.LEFT)
    with pytest.raises(RuleViolationError):
        service.add_goal(world.noa, Goal(trainee_id=world.yael.id, title="Anything", category="personal"))


def test_counselor_of_another_group_cannot_read_goals_or_history(world, service):
    with pytest.raises(NotFoundError, match="Trainee"):
        service.list_goals(world.dana, world.yael.id)
    with pytest.raises(NotFoundError, match=f"Goal {world.goal.id} does not exist"):
        service.history(world.dana, world.goal.id)


def test_a_stranger_learns_nothing_about_a_closed_file(world, service):
    """For a counselor of another group, a trainee who LEFT is a 404 like any other, not a 409
    "the file is closed": visibility is checked before anything that depends on the record."""
    world.trainees.update(world.yael.with_status(TraineeStatus.LEFT))
    with pytest.raises(NotFoundError):
        service.update_progress(world.dana, world.goal.id, 40, "Trying")
    with pytest.raises(RuleViolationError):  # who may see her gets the real reason
        service.update_progress(world.noa, world.goal.id, 40, "Trying")
