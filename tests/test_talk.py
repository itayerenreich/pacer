"""Domain rules of talks, talk proposals and notes: one object at a time, no database."""

from datetime import date

import pytest
from pydantic import ValidationError

from pacer.domain import (
    InvalidTransitionError,
    NoteKind,
    PersonalTalk,
    ProposalStatus,
    TalkProposal,
    TalkProposalKind,
    TalkStatus,
    TraineeNote,
)
from pacer.domain.errors import DomainRuleError

NOTES = "שיחה טובה, דיברנו על הריצה ועל היומן."


def talk(**changes) -> PersonalTalk:
    return PersonalTalk(**{"trainee_id": 1, "counselor_id": 2, "held_on": date(2026, 10, 13), "notes": NOTES, **changes})


def progress(**changes) -> TalkProposal:
    return TalkProposal(**{"talk_id": 1, "kind": "progress", "goal_id": 7, "progress": 60, "text": "רצה 3 ק״מ", **changes})


def note(**changes) -> TalkProposal:
    return TalkProposal(**{"talk_id": 1, "kind": "note", "note_kind": "worked", "text": "יעדים קטנים עובדים", **changes})


# --- PersonalTalk ---------------------------------------------------------------------------------


def test_a_talk_starts_recorded_and_moves_forward_with_its_summary():
    summarized = talk().advance(TalkStatus.SUMMARIZED, summary="סיכום", anonymized_notes="טקסט")
    assert summarized.status is TalkStatus.SUMMARIZED
    assert summarized.advance(TalkStatus.REVIEWED).status is TalkStatus.REVIEWED


def test_a_summarized_talk_must_have_its_summary_and_what_the_model_saw():
    with pytest.raises(ValidationError):
        talk().advance(TalkStatus.SUMMARIZED, summary="סיכום")
    with pytest.raises(ValidationError):
        talk(status=TalkStatus.REVIEWED)


def test_a_talk_cannot_skip_or_go_back():
    with pytest.raises(InvalidTransitionError):
        talk().advance(TalkStatus.REVIEWED)


def test_the_notes_never_show_in_repr():
    assert "ריצה" not in repr(talk())


# --- TalkProposal -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        {"goal_id": None},  # progress without a goal
        {"progress": None},
        {"note_kind": "worked"},  # progress with a note kind
        {"progress": 101},
    ],
)
def test_a_progress_proposal_has_exactly_its_fields(bad):
    with pytest.raises(ValidationError):
        progress(**bad)


@pytest.mark.parametrize("bad", [{"note_kind": None}, {"goal_id": 7}, {"progress": 50}, {"text": "א" * 301}])
def test_a_note_proposal_has_exactly_its_fields(bad):
    with pytest.raises(ValidationError):
        note(**bad)


def test_deciding_clears_the_text():
    approved = progress().approve(2, goal_update_id=9)
    rejected = note().reject(2)
    assert approved.text is None and approved.goal_update_id == 9 and approved.decided_by_id == 2
    assert rejected.text is None and rejected.status is ProposalStatus.REJECTED


def test_an_open_proposal_needs_its_text_and_a_decided_one_keeps_none():
    with pytest.raises(ValidationError):
        progress(text=None)
    approved = progress().approve(2, goal_update_id=9)
    with pytest.raises(ValidationError):
        TalkProposal.model_validate({**approved.model_dump(), "text": "הטקסט חזר"})


def test_an_approved_proposal_points_to_what_its_kind_makes():
    with pytest.raises(ValidationError):
        progress().approve(2)  # no GoalUpdate
    with pytest.raises(ValidationError):
        progress().approve(2, note_id=3)  # a progress proposal does not become a note
    with pytest.raises(ValidationError):
        note().approve(2, goal_update_id=3)


def test_an_approved_note_may_lose_its_note_when_the_counselor_deletes_it():
    approved = note().approve(2, note_id=3)
    assert TalkProposal.model_validate({**approved.model_dump(), "note_id": None}).note_id is None


def test_edits_depend_on_the_kind():
    assert progress().edited(progress=70, text="רצה 4 ק״מ").progress == 70
    assert note().edited(note_kind=NoteKind.FOLLOW_UP).note_kind is NoteKind.FOLLOW_UP
    with pytest.raises(DomainRuleError, match="note_kind"):
        progress().edited(note_kind=NoteKind.WORKED)
    with pytest.raises(DomainRuleError, match="progress"):
        note().edited(progress=50)
    with pytest.raises(DomainRuleError, match="status"):
        note().edited(status=ProposalStatus.APPROVED)


def test_a_decided_proposal_cannot_be_edited_or_decided_again():
    rejected = note().reject(2)
    with pytest.raises(DomainRuleError):
        rejected.edited(text="משהו אחר")
    with pytest.raises(InvalidTransitionError):
        rejected.approve(2, note_id=1)


# --- TraineeNote ---------------------------------------------------------------------------------------


def test_a_note_is_short():
    TraineeNote(trainee_id=1, kind=NoteKind.OBSERVATION, text="א" * 300, author_id=2)
    with pytest.raises(ValidationError):
        TraineeNote(trainee_id=1, kind=NoteKind.OBSERVATION, text="א" * 301, author_id=2)
    with pytest.raises(ValidationError):
        TraineeNote(trainee_id=1, kind=NoteKind.OBSERVATION, text="קצר", author_id=2)


def test_kinds_are_the_four_from_the_design():
    assert [k.value for k in NoteKind] == ["observation", "worked", "did_not_work", "follow_up"]
    assert [k.value for k in TalkProposalKind] == ["progress", "note"]
