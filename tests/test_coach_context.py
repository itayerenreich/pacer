"""CoachContextService: who may ask, what goes in (and the limits), and what never does."""

from datetime import UTC, datetime, timedelta

import pytest
from fakes import World

from pacer.domain import GoalUpdate, NoteKind, TraineeNote
from pacer.services import CoachContextService, NotFoundError
from pacer.services.coach_context import NOTES_LIMIT, UPDATES_PER_GOAL

START = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def service(world) -> CoachContextService:
    return CoachContextService(world.trainees, world.groups, world.goals, world.goal_updates, world.notes)


def add_note(world, text: str, days: int, kind: NoteKind = NoteKind.WORKED) -> TraineeNote:
    return world.notes.add(
        TraineeNote(
            trainee_id=world.yael.id,
            kind=kind,
            text=text,
            author_id=world.noa.id,
            created_at=START + timedelta(days=days),
        )
    )


def add_update(world, progress: int, days: int) -> None:
    world.goal_updates.add(
        GoalUpdate(
            goal_id=world.goal.id,
            author_id=world.noa.id,
            progress=progress,
            note=f"עדכון {progress}",
            created_at=START + timedelta(days=days),
        )
    )


# --- who may ask ---------------------------------------------------------------------------------


@pytest.mark.parametrize("who", ["manager", "noa", "omer"])
def test_everyone_who_sees_the_trainee_may_ask(world, service, who):
    """The manager, the primary counselor, and the group's second counselor (decision 2026-10-25)."""
    context = service.build(getattr(world, who), world.yael.id)
    assert context.trainee == world.yael


def test_a_counselor_of_another_group_gets_404_as_if_the_trainee_did_not_exist(world, service):
    with pytest.raises(NotFoundError) as hidden:
        service.build(world.dana, world.yael.id)
    with pytest.raises(NotFoundError) as missing:
        service.build(world.manager, 999)
    assert str(hidden.value).replace(str(world.yael.id), "N") == str(missing.value).replace("999", "N")


# --- what goes in --------------------------------------------------------------------------------


def test_goals_with_their_most_recent_updates_oldest_first(world, service):
    for i, progress in enumerate([10, 20, 30, 40, 50]):
        add_update(world, progress, days=i)
    (goal,) = service.build(world.noa, world.yael.id).goals
    assert goal.goal == world.goal
    assert [u.progress for u in goal.recent_updates] == [30, 40, 50]
    assert len(goal.recent_updates) == UPDATES_PER_GOAL


def test_only_the_most_recent_notes_oldest_first(world, service):
    for day in range(NOTES_LIMIT + 3):
        add_note(world, f"הערה מיום {day}", days=day)
    notes = service.build(world.noa, world.yael.id).notes
    assert len(notes) == NOTES_LIMIT
    assert notes[0].text == "הערה מיום 3" and notes[-1].text == f"הערה מיום {NOTES_LIMIT + 2}"


def test_a_trainee_with_nothing_yet_still_has_a_context(world, service):
    world.goals.rows.clear()
    context = service.build(world.noa, world.yael.id)
    assert context.goals == [] and context.notes == []


def test_another_trainees_notes_never_come_in(world, service):
    other = world.trainees.add(world.yael.model_copy(update={"id": None, "first_name": "Other"}))
    world.notes.add(
        TraineeNote(trainee_id=other.id, kind=NoteKind.WORKED, text="של חניך אחר", author_id=world.noa.id)
    )
    assert service.build(world.noa, world.yael.id).notes == []


def test_the_personal_file_is_never_part_of_it(world, service):
    """By construction: the service is not even given the personal file repository."""
    context = service.build(world.manager, world.yael.id)
    assert "Guitar" not in repr(context)
    assert "personal_files" not in CoachContextService.__init__.__code__.co_varnames
