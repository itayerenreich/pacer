"""Note retention: a closed file starts a clock, and the notes are deleted when it runs out.

Domain: Trainee.closed_at follows the status. Service: which notes are past the period, and the
delete. Command line: shows by default, deletes only with --yes.
"""

from datetime import UTC, date, datetime, timedelta

import pytest
from fakes import World

from pacer.domain import NoteKind, PersonalTalk, TalkProposal, TalkProposalKind, Trainee, TraineeNote, TraineeStatus
from pacer.domain.errors import DomainRuleError
from pacer.services.retention import DEFAULT_KEEP_FOR, MAX_KEEP_FOR, past_retention, purge

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
    talk = world.talks.add(PersonalTalk(trainee_id=world.yael.id, counselor_id=world.noa.id, held_on=date(2026, 10, 1), notes="כאבי ראש, הולכת לרופא."))
    world.talk_proposals.add(TalkProposal(talk_id=talk.key, kind=TalkProposalKind.NOTE, note_kind=NoteKind.FOLLOW_UP, text="לבדוק איך היה"))
    return world


def close(world: World, at: datetime) -> None:
    world.trainees.update(world.yael.with_status(Status.LEFT, at=at))


def plan_for(world: World, now: datetime = NOW, keep_for: timedelta = DEFAULT_KEEP_FOR):
    return past_retention(world.trainees, world.notes, world.talks, now=now, keep_for=keep_for)


def run_purge(world: World, plan, now: datetime = NOW):
    return purge(world.trainees, world.notes, world.talks, plan, now=now)


def test_an_open_file_is_never_purged(world):
    plan = plan_for(world, now=NOW + timedelta(days=10_000))
    assert (plan.note_count, plan.talk_count) == (0, 0)


def test_records_are_kept_until_the_period_is_over(world):
    close(world, at=NOW - DEFAULT_KEEP_FOR)  # exactly the period: still kept
    assert plan_for(world).trainee_ids == set()


def test_notes_and_talks_past_the_period_are_deleted(world):
    """Review 2026-10-18: the talks (the counselor's raw notes, with the sentences about health
    that are kept locally) must go too, with every proposal made from them."""
    close(world, at=NOW - DEFAULT_KEEP_FOR - timedelta(days=1))
    plan = plan_for(world)
    assert plan.notes_by_trainee == {world.yael.id: [1, 2]} and plan.talk_count == 1
    done = run_purge(world, plan)
    assert (done.note_count, done.talk_count) == (2, 1)
    assert world.notes.list_by_trainee(world.yael.id) == []
    assert world.talks.list_by_trainee(world.yael.id) == []
    assert world.talk_proposals.rows == {}


def test_a_file_reopened_after_the_plan_was_made_is_skipped(world):
    close(world, at=NOW - DEFAULT_KEEP_FOR - timedelta(days=1))
    plan = plan_for(world)
    world.trainees.update(world.trainees.get(world.yael.id).with_status(Status.ACTIVE))  # came back
    assert run_purge(world, plan).trainee_ids == set()
    assert len(world.notes.list_by_trainee(world.yael.id)) == 2


def test_the_period_must_be_sensible(world):
    close(world, at=NOW - timedelta(days=31))
    assert plan_for(world, keep_for=timedelta(days=30)).note_count == 2
    for bad in (timedelta(days=-1), MAX_KEEP_FOR + timedelta(days=1)):
        with pytest.raises(ValueError):
            plan_for(world, keep_for=bad)


@pytest.mark.parametrize("path", ["manager", "approval"])
def test_both_ways_to_change_a_status_close_the_file(world, path):
    from pacer.domain import ApprovalKind
    from pacer.services import ApprovalService, TraineeService

    if path == "manager":
        service = TraineeService(world.trainees, world.groups, world.personal_files, world.approvals)
        service.change_status(world.manager, world.yael.id, Status.LEFT)
    else:
        approvals = ApprovalService(world.approvals, world.trainees, world.groups)
        request = approvals.request(world.noa, ApprovalKind.STATUS_CHANGE, world.yael.id, "עזבה", Status.LEFT)
        approvals.approve(world.manager, request.key)
    assert world.trainees.get(world.yael.id).closed_at is not None
