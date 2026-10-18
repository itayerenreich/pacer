"""Note retention: a closed file starts a clock, and the notes are deleted when it runs out.

Domain: Trainee.closed_at follows the status. Service: which notes are past the period, and the
delete. Command line: shows by default, deletes only with --yes.
"""

from datetime import UTC, datetime, timedelta

import pytest
from fakes import World

from pacer.domain import NoteKind, Trainee, TraineeNote, TraineeStatus
from pacer.domain.errors import DomainRuleError
from pacer.services.retention import DEFAULT_KEEP_NOTES_FOR, notes_past_retention, purge_notes

NOW = datetime(2027, 10, 1, tzinfo=UTC)
Status = TraineeStatus


def trainee(**changes) -> Trainee:
    return Trainee(first_name="יעל", last_name="מזרחי", cohort_id=1, group_id=1, primary_counselor_id=1, **changes)


# --- the domain: closed_at follows the status ------------------------------------------------------


def test_closing_a_file_records_when():
    closed = trainee().with_status(Status.LEFT, at=NOW)
    assert closed.closed_at == NOW


def test_moving_between_closed_statuses_keeps_the_first_date():
    closed = trainee().with_status(Status.LEFT, at=NOW)
    assert closed.with_status(Status.GRADUATED, at=NOW + timedelta(days=30)).closed_at == NOW


@pytest.mark.parametrize("status", [Status.ACTIVE, Status.ON_LEAVE])
def test_reopening_a_file_clears_the_date(status):
    assert trainee().with_status(Status.LEFT, at=NOW).with_status(status).closed_at is None


def test_closing_without_a_date_uses_now():
    before = datetime.now(UTC)
    assert trainee().with_status(Status.GRADUATED).closed_at >= before


def test_a_closed_trainee_without_a_date_and_an_open_one_with_a_date_are_invalid():
    with pytest.raises(DomainRuleError):
        trainee(status=Status.LEFT)
    with pytest.raises(DomainRuleError):
        trainee(closed_at=NOW)


# --- the service ------------------------------------------------------------------------------------


@pytest.fixture
def world() -> World:
    world = World()
    for text in ("יעד קטן ומדיד עוזר לה", "תזכורות בווטסאפ לא עובדות"):
        world.notes.add(TraineeNote(trainee_id=world.yael.id, kind=NoteKind.WORKED, text=text, author_id=world.noa.id))
    return world


def close(world: World, at: datetime) -> None:
    world.trainees.update(world.yael.with_status(Status.LEFT, at=at))


def plan_for(world: World, now: datetime = NOW):
    return notes_past_retention(world.trainees, world.notes, now=now)


def test_notes_of_an_open_file_are_never_purged(world):
    assert plan_for(world, now=NOW + timedelta(days=10_000)).note_count == 0


def test_notes_are_kept_until_the_period_is_over(world):
    close(world, at=NOW - DEFAULT_KEEP_NOTES_FOR)  # exactly the period: still kept
    assert plan_for(world).note_count == 0


def test_notes_past_the_period_are_deleted(world):
    close(world, at=NOW - DEFAULT_KEEP_NOTES_FOR - timedelta(days=1))
    plan = plan_for(world)
    assert plan.notes_by_trainee == {world.yael.id: [1, 2]}
    assert purge_notes(world.notes, plan) == 2
    assert world.notes.list_by_trainee(world.yael.id) == []


def test_a_shorter_period_can_be_chosen_and_a_negative_one_cannot(world):
    close(world, at=NOW - timedelta(days=31))
    assert notes_past_retention(world.trainees, world.notes, now=NOW, keep_for=timedelta(days=30)).note_count == 2
    with pytest.raises(ValueError):
        notes_past_retention(world.trainees, world.notes, now=NOW, keep_for=timedelta(days=-1))


def test_the_status_change_through_the_services_closes_the_file(world):
    """Both ways a status changes (manager directly, or an approved request) go through
    Trainee.with_status, so both record closed_at."""
    from pacer.services import TraineeService

    service = TraineeService(world.trainees, world.groups, world.personal_files, world.approvals)
    closed = service.change_status(world.manager, world.yael.id, Status.LEFT)
    assert closed.closed_at is not None
