"""TalkService with in-memory fakes: permissions, approvals into GoalUpdates and notes, and memory."""

from datetime import date

import pytest
from fakes import World
from pydantic import ValidationError

from pacer.domain import (
    DomainRuleError,
    GoalStatus,
    InvalidTransitionError,
    NoteKind,
    ProposalStatus,
    TalkProposalKind,
    TalkStatus,
    TraineeStatus,
)
from pacer.services import (
    GoalService,
    PermissionDeniedError,
    RuleViolationError,
    TalkDraft,
    TalkProposalEdits,
    TalkService,
)

NOTES = "דיברנו על הריצה. רצה 3 ק״מ ברצף. יעדים קטנים עובדים לה."


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def service(world) -> TalkService:
    goals = GoalService(world.goals, world.goal_updates, world.trainees, world.groups)
    return TalkService(
        world.talks, world.talk_proposals, world.notes, world.trainees, world.groups, world.goals, goals
    )


def summarized_talk(world, service, drafts: list[TalkDraft]):
    """A talk as the pipeline leaves it: summarized, with proposals."""
    talk = service.record_talk(world.noa, world.yael.id, held_on=date(2026, 10, 13), notes=NOTES)
    world.talks.update(talk.advance(TalkStatus.SUMMARIZED, summary="סיכום", anonymized_notes="טקסט"))
    return talk, service.submit_proposals(talk.id, drafts)


PROGRESS = TalkDraft(kind=TalkProposalKind.PROGRESS, goal_id=1, progress=60, text="רצה 3 ק״מ ברצף")
NOTE = TalkDraft(kind=TalkProposalKind.NOTE, note_kind=NoteKind.WORKED, text="יעדים קטנים ומדידים עובדים")


# --- recording and viewing ---------------------------------------------------------------------------


def test_the_primary_counselor_records_and_both_group_counselors_see(world, service):
    talk = service.record_talk(world.noa, world.yael.id, held_on=date(2026, 10, 13), notes=NOTES)
    assert talk.counselor_id == world.noa.id and talk.status is TalkStatus.RECORDED
    assert service.get_talk(world.omer, talk.id) == talk  # Omer leads the group too: may read
    assert service.list_talks(world.manager, world.yael.id) == [talk]


def test_the_other_group_counselor_cannot_record(world, service):
    with pytest.raises(PermissionDeniedError):
        service.record_talk(world.omer, world.yael.id, held_on=date(2026, 10, 13), notes=NOTES)


def test_a_counselor_of_another_group_sees_nothing(world, service):
    talk = service.record_talk(world.noa, world.yael.id, held_on=date(2026, 10, 13), notes=NOTES)
    for action in (
        lambda: service.get_talk(world.dana, talk.id),
        lambda: service.list_talks(world.dana, world.yael.id),
        lambda: service.list_notes(world.dana, world.yael.id),
        lambda: service.list_proposals(world.dana, talk.id),
    ):
        with pytest.raises(PermissionDeniedError):
            action()


def test_no_talks_with_a_trainee_who_left(world, service):
    world.trainees.update(world.yael.model_copy(update={"status": TraineeStatus.LEFT}))
    with pytest.raises(RuleViolationError, match="closed"):
        service.record_talk(world.noa, world.yael.id, held_on=date(2026, 10, 13), notes=NOTES)


# --- proposals ----------------------------------------------------------------------------------------


def test_proposals_only_come_with_a_summary(world, service):
    talk = service.record_talk(world.noa, world.yael.id, held_on=date(2026, 10, 13), notes=NOTES)
    with pytest.raises(RuleViolationError, match="with the summary"):
        service.submit_proposals(talk.id, [NOTE])


def test_approving_progress_is_a_regular_goal_update(world, service):
    _, [proposal] = summarized_talk(world, service, [PROGRESS])
    approved = service.approve_proposal(world.noa, proposal.id)

    goal = world.goals.get(1)
    [update] = world.goal_updates.list_by_goal(1)
    assert (goal.progress, goal.status) == (60, GoalStatus.IN_PROGRESS)
    assert (update.progress, update.note, update.author_id) == (60, "רצה 3 ק״מ ברצף", world.noa.id)
    assert approved.goal_update_id == update.id and approved.text is None


def test_approving_a_note_adds_it_to_the_trainees_memory_by_the_approver(world, service):
    talk, [proposal] = summarized_talk(world, service, [NOTE])
    approved = service.approve_proposal(world.noa, proposal.id, TalkProposalEdits(text="יעדים קטנים עובדים לה"))

    [note] = service.list_notes(world.omer, world.yael.id)
    assert (note.text, note.kind, note.author_id, note.source_talk_id) == (
        "יעדים קטנים עובדים לה",
        NoteKind.WORKED,
        world.noa.id,
        talk.id,
    )
    assert approved.note_id == note.id


def test_rejecting_keeps_no_text_and_creates_nothing(world, service):
    _, [proposal] = summarized_talk(world, service, [NOTE])
    rejected = service.reject_proposal(world.noa, proposal.id)
    assert rejected.status is ProposalStatus.REJECTED and rejected.text is None
    assert world.talk_proposals.get(proposal.id).text is None  # not in storage either
    assert service.list_notes(world.noa, world.yael.id) == []


def test_only_the_primary_counselor_or_the_manager_decides(world, service):
    _, [proposal] = summarized_talk(world, service, [PROGRESS])
    with pytest.raises(PermissionDeniedError):
        service.approve_proposal(world.omer, proposal.id)
    with pytest.raises(PermissionDeniedError):
        service.reject_proposal(world.dana, proposal.id)
    assert service.approve_proposal(world.manager, proposal.id).status is ProposalStatus.APPROVED


def test_deciding_twice_creates_nothing_the_second_time(world, service):
    _, [proposal] = summarized_talk(world, service, [NOTE])
    service.approve_proposal(world.noa, proposal.id)
    with pytest.raises(InvalidTransitionError):
        service.approve_proposal(world.noa, proposal.id)
    with pytest.raises(InvalidTransitionError):
        service.reject_proposal(world.noa, proposal.id)
    assert len(service.list_notes(world.noa, world.yael.id)) == 1


def test_a_goal_of_another_trainee_is_refused(world, service):
    other = world.goals.add(world.goal.model_copy(update={"id": None, "trainee_id": 99}))
    draft = TalkDraft(kind=TalkProposalKind.PROGRESS, goal_id=other.id, progress=50, text="משהו שקרה")
    _, [proposal] = summarized_talk(world, service, [draft])
    with pytest.raises(RuleViolationError, match="not a goal of this talk's trainee"):
        service.approve_proposal(world.noa, proposal.id)
    assert world.goal_updates.list_by_goal(other.id) == []


def test_a_dropped_goal_refuses_progress_at_approval(world, service):
    _, [proposal] = summarized_talk(world, service, [PROGRESS])
    world.goals.update(world.goal.model_copy(update={"status": GoalStatus.DROPPED}))
    with pytest.raises(DomainRuleError, match="dropped"):
        service.approve_proposal(world.noa, proposal.id)
    assert world.talk_proposals.get(proposal.id).status is ProposalStatus.PROPOSED


def test_a_note_edited_to_be_too_long_is_refused(world, service):
    _, [proposal] = summarized_talk(world, service, [NOTE])
    with pytest.raises(ValidationError, match="at most 300"):
        service.approve_proposal(world.noa, proposal.id, TalkProposalEdits(text="א" * 400))
    assert service.list_notes(world.noa, world.yael.id) == []


def test_the_talk_is_reviewed_when_every_proposal_is_decided(world, service):
    talk, [progress, note] = summarized_talk(world, service, [PROGRESS, NOTE])
    service.approve_proposal(world.noa, progress.id)
    assert world.talks.get(talk.id).status is TalkStatus.SUMMARIZED
    service.reject_proposal(world.noa, note.id)
    assert world.talks.get(talk.id).status is TalkStatus.REVIEWED


def test_a_talk_without_proposals_is_reviewed_at_once(world, service):
    talk, proposals = summarized_talk(world, service, [])
    assert proposals == [] and world.talks.get(talk.id).status is TalkStatus.REVIEWED


# --- notes by hand, and deleting -------------------------------------------------------------------


def test_notes_by_hand_follow_the_same_rules(world, service):
    note = service.add_note(world.noa, world.yael.id, kind=NoteKind.FOLLOW_UP, text="לחזור לנושא הטיול")
    assert note.source_talk_id is None and note.author_id == world.noa.id
    with pytest.raises(PermissionDeniedError):
        service.add_note(world.omer, world.yael.id, kind=NoteKind.OBSERVATION, text="משהו שראיתי")


def test_deleting_a_note_really_deletes_it(world, service):
    _, [proposal] = summarized_talk(world, service, [NOTE])
    note_id = service.approve_proposal(world.noa, proposal.id).note_id

    with pytest.raises(PermissionDeniedError):
        service.delete_note(world.omer, note_id)
    service.delete_note(world.noa, note_id)

    assert service.list_notes(world.noa, world.yael.id) == []
    stored = world.talk_proposals.get(proposal.id)  # still a valid record, pointing nowhere, no text
    assert (stored.status, stored.note_id, stored.text) == (ProposalStatus.APPROVED, None, None)


def test_a_note_can_be_deleted_after_the_trainee_left(world, service):
    note = service.add_note(world.noa, world.yael.id, kind=NoteKind.OBSERVATION, text="משהו שחשוב לזכור")
    world.trainees.update(world.yael.model_copy(update={"status": TraineeStatus.GRADUATED}))
    service.delete_note(world.noa, note.id)
    assert world.notes.get(note.id) is None


def test_the_note_belongs_to_whoever_approved_it(world, service):
    """Noa held the talk; the manager approved the note, so the manager is its author."""
    _, [proposal] = summarized_talk(world, service, [NOTE])
    service.approve_proposal(world.manager, proposal.id)
    [note] = service.list_notes(world.noa, world.yael.id)
    assert note.author_id == world.manager.id
