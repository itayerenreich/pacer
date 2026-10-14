"""Tests for the ApprovalRequest state machine (domain only, no database)."""

import pytest
from pydantic import ValidationError

from pacer.domain import ApprovalKind, ApprovalRequest, ApprovalStatus, InvalidTransitionError

MANAGER_ID = 1


def make_request(**overrides) -> ApprovalRequest:
    data = {
        "kind": ApprovalKind.VIEW_PERSONAL_FILE,
        "trainee_id": 10,
        "requested_by_id": 2,
        "reason": "Preparing for a personal talk",
    }
    data.update(overrides)
    return ApprovalRequest(**data)


def test_new_request_is_pending_without_a_decision():
    request = make_request()
    assert request.status is ApprovalStatus.PENDING
    assert request.decided_by_id is None
    assert request.decided_at is None


def test_approve_returns_a_new_decided_request():
    request = make_request()
    approved = request.approve(MANAGER_ID, note="OK, for this talk")

    assert approved.status is ApprovalStatus.APPROVED
    assert approved.decided_by_id == MANAGER_ID
    assert approved.decided_at.utcoffset() is not None
    assert request.status is ApprovalStatus.PENDING  # the original did not change


def test_reject_moves_to_rejected():
    assert make_request().reject(MANAGER_ID).status is ApprovalStatus.REJECTED


@pytest.mark.parametrize("decide", ["approve", "reject"])
@pytest.mark.parametrize("first", ["approve", "reject"])
def test_a_decided_request_cannot_be_decided_again(first, decide):
    decided = getattr(make_request(), first)(MANAGER_ID)
    with pytest.raises(InvalidTransitionError):
        getattr(decided, decide)(MANAGER_ID)


def test_status_change_must_name_the_new_status():
    with pytest.raises(ValidationError, match="must say which status"):
        make_request(kind=ApprovalKind.STATUS_CHANGE)


def test_only_status_change_has_a_requested_status():
    with pytest.raises(ValidationError, match="Only a status change"):
        make_request(requested_status="left")


def test_cannot_build_an_approved_request_without_a_decider():
    with pytest.raises(ValidationError, match="who decided and when"):
        make_request(status="approved")
