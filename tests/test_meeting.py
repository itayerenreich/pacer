"""The three state machines of meetings, proposals and tasks (domain only)."""

from datetime import date

import pytest
from pydantic import ValidationError

from pacer.domain import (
    DomainRuleError,
    InvalidTransitionError,
    MeetingStatus,
    ProposalStatus,
    ProposedTask,
    StaffMeeting,
    Task,
    TaskStatus,
)

TRANSCRIPT = "נועה: נתחיל בעדכונים על הקבוצה. עומר: השבוע היה טוב."
M = MeetingStatus


def meeting(**overrides) -> StaffMeeting:
    data = {
        "title": "ישיבת צוות שבועית",
        "held_on": date(2026, 10, 12),
        "created_by_id": 1,
        "participant_ids": [1, 2, 3],
    }
    data.update(overrides)
    return StaffMeeting(**data)


# --- StaffMeeting --------------------------------------------------------------------------


def test_a_meeting_moves_through_the_pipeline_one_step_at_a_time():
    m = meeting(status=M.TRANSCRIBED, transcript=TRANSCRIPT)
    m = m.advance(M.ANONYMIZED, anonymized_transcript="[STAFF_2]: נתחיל")
    m = m.advance(M.EXTRACTED, summary="סיכום")
    m = m.advance(M.PENDING_APPROVAL)
    m = m.advance(M.APPROVED)
    assert m.status is M.APPROVED


def test_a_step_cannot_be_skipped():
    with pytest.raises(InvalidTransitionError, match="from transcribed to extracted"):
        meeting(status=M.TRANSCRIBED, transcript=TRANSCRIPT).advance(M.EXTRACTED, summary="קצר")


def test_each_status_requires_its_field():
    """You cannot claim 'anonymized' without the anonymized text, or 'extracted' without a summary."""
    with pytest.raises(ValidationError, match="must have a transcript"):
        meeting(status=M.TRANSCRIBED)
    with pytest.raises(ValidationError, match="must have a anonymized_transcript"):
        meeting(status=M.TRANSCRIBED, transcript=TRANSCRIPT).advance(M.ANONYMIZED)


def test_advance_returns_a_new_meeting():
    original = meeting(status=M.TRANSCRIBED, transcript=TRANSCRIPT)
    original.advance(M.ANONYMIZED, anonymized_transcript="טקסט")
    assert original.status is M.TRANSCRIBED


def test_participants_are_unique_and_at_least_one():
    with pytest.raises(ValidationError):
        meeting(participant_ids=[])
    with pytest.raises(ValidationError, match="more than once"):
        meeting(participant_ids=[2, 2])


def test_the_transcript_never_shows_in_repr():
    assert "נועה" not in repr(meeting(status=M.TRANSCRIBED, transcript=TRANSCRIPT))


# --- ProposedTask ----------------------------------------------------------------------------


def proposal(**overrides) -> ProposedTask:
    data = {"meeting_id": 1, "description": "לתאם הסעה לטיול", "owner_id": 2}
    data.update(overrides)
    return ProposedTask(**data)


def test_approve_records_who_when_and_the_task():
    approved = proposal().approve(manager_id=1, task_id=10)
    assert (approved.status, approved.decided_by_id, approved.task_id) == (ProposalStatus.APPROVED, 1, 10)
    assert approved.decided_at is not None


def test_a_proposal_is_decided_once():
    with pytest.raises(InvalidTransitionError, match="already rejected"):
        proposal().reject(1).approve(1, task_id=10)


def test_cannot_approve_without_an_owner():
    with pytest.raises(ValidationError, match="must have an owner"):
        proposal(owner_id=None).approve(1, task_id=10)


def test_edits_change_only_the_three_editable_fields():
    edited = proposal().edited(description="לתאם הסעה ליום שלישי", owner_id=3)
    assert (edited.description, edited.owner_id) == ("לתאם הסעה ליום שלישי", 3)
    with pytest.raises(ValueError, match="Cannot edit"):
        proposal().edited(status="approved")


def test_a_decided_proposal_cannot_be_edited():
    with pytest.raises(DomainRuleError):
        proposal().reject(1).edited(owner_id=3)


def test_edits_are_validated_like_everything_else():
    with pytest.raises(ValidationError):
        proposal().edited(description="קצר")


# --- Task -------------------------------------------------------------------------------------


def test_complete_once():
    done = Task(description="לתאם הסעה", owner_id=2, created_by_id=1).complete()
    assert done.status is TaskStatus.DONE and done.completed_at is not None
    with pytest.raises(InvalidTransitionError, match="already done"):
        done.complete()
