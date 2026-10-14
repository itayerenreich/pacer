"""TraineeService tests with in-memory fakes: enrolling, viewing, personal files, status."""

import pytest
from fakes import World

from pacer.domain import ApprovalKind, Trainee, TraineeStatus
from pacer.services import (
    ApprovalService,
    PermissionDeniedError,
    RuleViolationError,
    TraineeService,
)


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def service(world) -> TraineeService:
    return TraineeService(world.trainees, world.groups, world.personal_files, world.approvals)


@pytest.fixture
def approvals(world) -> ApprovalService:
    return ApprovalService(world.approvals, world.trainees, world.groups)


def new_trainee(world: World, **overrides) -> Trainee:
    data = {
        "first_name": "Itamar",
        "last_name": "Peretz",
        "cohort_id": world.cohort.id,
        "group_id": world.group_a.id,
        "primary_counselor_id": world.omer.id,
    }
    data.update(overrides)
    return Trainee(**data)


# --- enroll ------------------------------------------------------------------------


def test_manager_enrolls_a_trainee(world, service):
    trainee = service.enroll(world.manager, new_trainee(world))
    assert world.trainees.get(trainee.id).full_name == "Itamar Peretz"


def test_counselor_cannot_enroll(world, service):
    with pytest.raises(PermissionDeniedError):
        service.enroll(world.noa, new_trainee(world))


def test_primary_counselor_must_lead_the_group(world, service):
    with pytest.raises(RuleViolationError, match="does not lead group"):
        service.enroll(world.manager, new_trainee(world, primary_counselor_id=world.dana.id))


def test_group_must_be_in_the_same_cohort(world, service):
    with pytest.raises(RuleViolationError, match="belongs to cohort"):
        service.enroll(world.manager, new_trainee(world, cohort_id=world.cohort.id + 1))


# --- get ---------------------------------------------------------------------------


def test_group_counselors_see_the_trainee_others_do_not(world, service):
    assert service.get(world.omer, world.yael.id) == world.yael
    with pytest.raises(PermissionDeniedError):
        service.get(world.dana, world.yael.id)


# --- personal file -----------------------------------------------------------------


def test_manager_always_sees_the_personal_file(world, service):
    assert service.get_personal_file(world.manager, world.yael.id).intake_answers == {"Hobbies": "Guitar"}


def test_counselor_needs_an_approved_request(world, service, approvals):
    with pytest.raises(PermissionDeniedError, match="without manager approval"):
        service.get_personal_file(world.noa, world.yael.id)

    request = approvals.request(
        world.noa, ApprovalKind.VIEW_PERSONAL_FILE, world.yael.id, "Personal talk tomorrow"
    )
    with pytest.raises(PermissionDeniedError):  # pending is not enough
        service.get_personal_file(world.noa, world.yael.id)

    approvals.approve(world.manager, request.id)
    assert service.get_personal_file(world.noa, world.yael.id).trainee_id == world.yael.id


def test_approval_is_personal_not_shared_with_the_co_counselor(world, service, approvals):
    request = approvals.request(world.noa, ApprovalKind.VIEW_PERSONAL_FILE, world.yael.id, "Personal talk")
    approvals.approve(world.manager, request.id)
    with pytest.raises(PermissionDeniedError):
        service.get_personal_file(world.omer, world.yael.id)


def test_access_ends_when_the_counselor_no_longer_leads_the_group(world, service, approvals):
    request = approvals.request(world.noa, ApprovalKind.VIEW_PERSONAL_FILE, world.yael.id, "Personal talk")
    approvals.approve(world.manager, request.id)
    world.groups.update(world.group_a.model_copy(update={"counselor_ids": [world.omer.id]}))  # Noa moved

    with pytest.raises(PermissionDeniedError):
        service.get_personal_file(world.noa, world.yael.id)


def test_rejected_request_gives_no_access(world, service, approvals):
    request = approvals.request(world.noa, ApprovalKind.VIEW_PERSONAL_FILE, world.yael.id, "Curious")
    approvals.reject(world.manager, request.id, note="No real need")
    with pytest.raises(PermissionDeniedError):
        service.get_personal_file(world.noa, world.yael.id)


# --- status ------------------------------------------------------------------------


def test_manager_changes_status_directly(world, service):
    assert (
        service.change_status(world.manager, world.yael.id, TraineeStatus.ON_LEAVE).status
        is TraineeStatus.ON_LEAVE
    )
    assert world.trainees.get(world.yael.id).status is TraineeStatus.ON_LEAVE


def test_counselor_cannot_change_status_directly(world, service):
    with pytest.raises(PermissionDeniedError, match="ask the manager"):
        service.change_status(world.noa, world.yael.id, TraineeStatus.LEFT)
    assert world.trainees.get(world.yael.id).status is TraineeStatus.ACTIVE
