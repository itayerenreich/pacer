"""ApprovalService tests with in-memory fakes: asking, approving, rejecting."""

import pytest
from fakes import World

from pacer.domain import ApprovalKind, ApprovalStatus, InvalidTransitionError, TraineeStatus
from pacer.services import ApprovalService, NotFoundError, PermissionDeniedError, RuleViolationError

VIEW = ApprovalKind.VIEW_PERSONAL_FILE
STATUS = ApprovalKind.STATUS_CHANGE


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def service(world) -> ApprovalService:
    return ApprovalService(world.approvals, world.trainees, world.groups)


# --- request -----------------------------------------------------------------------


def test_group_counselor_can_ask(world, service):
    request = service.request(world.omer, VIEW, world.yael.id, "Preparing a talk")
    assert request.status is ApprovalStatus.PENDING
    assert request.requested_by_id == world.omer.id


def test_counselor_of_another_group_cannot_ask(world, service):
    with pytest.raises(NotFoundError):  # a trainee they may not see does not exist for them
        service.request(world.dana, VIEW, world.yael.id, "Just looking")


def test_manager_does_not_ask(world, service):
    with pytest.raises(RuleViolationError, match="does not need approval"):
        service.request(world.manager, VIEW, world.yael.id, "Self-approval")


def test_no_duplicate_pending_requests(world, service):
    service.request(world.noa, VIEW, world.yael.id, "Talk on Sunday")
    with pytest.raises(RuleViolationError, match="already have a pending"):
        service.request(world.noa, VIEW, world.yael.id, "Asking again")


def test_no_new_request_when_access_was_already_approved(world, service):
    first = service.request(world.noa, VIEW, world.yael.id, "Talk on Sunday")
    service.approve(world.manager, first.id)
    with pytest.raises(RuleViolationError, match="already have access"):
        service.request(world.noa, VIEW, world.yael.id, "Again")


def test_can_ask_again_after_a_rejection(world, service):
    first = service.request(world.noa, VIEW, world.yael.id, "Curious")
    service.reject(world.manager, first.id)
    assert service.request(world.noa, VIEW, world.yael.id, "A real reason this time").is_pending


def test_status_change_to_the_current_status_is_pointless(world, service):
    with pytest.raises(RuleViolationError, match="already active"):
        service.request(world.noa, STATUS, world.yael.id, "No change", requested_status=TraineeStatus.ACTIVE)


def test_missing_trainee_is_not_found(world, service):
    with pytest.raises(NotFoundError):
        service.request(world.noa, VIEW, 999, "Who?")


# --- approve / reject --------------------------------------------------------------


def test_approving_a_status_change_applies_it(world, service):
    request = service.request(
        world.noa, STATUS, world.yael.id, "Sent home for 3 days", requested_status="on_leave"
    )
    assert world.trainees.get(world.yael.id).status is TraineeStatus.ACTIVE  # nothing happens before approval

    approved = service.approve(world.manager, request.id, note="Agreed")

    assert approved.status is ApprovalStatus.APPROVED
    assert world.approvals.get(request.id).decided_by_id == world.manager.id
    assert world.trainees.get(world.yael.id).status is TraineeStatus.ON_LEAVE


def test_rejecting_a_status_change_changes_nothing(world, service):
    request = service.request(world.noa, STATUS, world.yael.id, "Wants to leave", requested_status="left")
    service.reject(world.manager, request.id, note="Talk with her first")

    assert world.approvals.get(request.id).status is ApprovalStatus.REJECTED
    assert world.trainees.get(world.yael.id).status is TraineeStatus.ACTIVE


def test_only_the_manager_decides(world, service):
    request = service.request(world.noa, VIEW, world.yael.id, "Talk")
    with pytest.raises(PermissionDeniedError):
        service.approve(world.omer, request.id)
    with pytest.raises(PermissionDeniedError):
        service.reject(world.noa, request.id)


def test_a_request_is_decided_only_once(world, service):
    request = service.request(world.noa, VIEW, world.yael.id, "Talk")
    service.reject(world.manager, request.id)
    with pytest.raises(InvalidTransitionError):
        service.approve(world.manager, request.id)
    assert world.approvals.get(request.id).status is ApprovalStatus.REJECTED


# --- list_pending ------------------------------------------------------------------


def test_manager_sees_all_pending_counselors_see_their_own(world, service):
    noa_request = service.request(world.noa, VIEW, world.yael.id, "Talk")
    omer_request = service.request(world.omer, VIEW, world.yael.id, "Another talk")
    decided = service.request(world.noa, STATUS, world.yael.id, "Leave", requested_status="on_leave")
    service.reject(world.manager, decided.id)

    assert [r.id for r in service.list_pending(world.manager)] == [noa_request.id, omer_request.id]
    assert [r.id for r in service.list_pending(world.omer)] == [omer_request.id]
