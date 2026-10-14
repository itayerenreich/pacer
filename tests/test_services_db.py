"""Integration tests: services + real repositories + PostgreSQL (see conftest.py).

The fake-based tests check the rules. These check what fakes cannot: that the new
approval_requests migration and repository work, and that a service's writes are
saved together or not at all inside one unit of work.
"""

import pytest
from sqlalchemy.orm import Session
from test_repositories import make_trainee, seed_group

from pacer.domain import (
    ApprovalKind,
    ApprovalStatus,
    Goal,
    GoalStatus,
    PersonalFile,
    StaffMember,
    TraineeStatus,
)
from pacer.repositories import (
    ApprovalRequestRepository,
    GoalRepository,
    GoalUpdateRepository,
    GroupRepository,
    PersonalFileRepository,
    StaffRepository,
    TraineeRepository,
    unit_of_work,
)
from pacer.services import ApprovalService, GoalService, TraineeService

pytestmark = pytest.mark.integration


def goal_service(session: Session) -> GoalService:
    return GoalService(
        GoalRepository(session),
        GoalUpdateRepository(session),
        TraineeRepository(session),
        GroupRepository(session),
    )


def approval_service(session: Session) -> ApprovalService:
    return ApprovalService(
        ApprovalRequestRepository(session), TraineeRepository(session), GroupRepository(session)
    )


def trainee_service(session: Session) -> TraineeService:
    return TraineeService(
        TraineeRepository(session),
        GroupRepository(session),
        PersonalFileRepository(session),
        ApprovalRequestRepository(session),
    )


@pytest.fixture
def academy(engine):
    """A group with two counselors, a manager, and one trainee with a goal and a personal file."""
    seed = seed_group(engine)
    with unit_of_work(engine) as session:
        staff = StaffRepository(session)
        manager = staff.add(StaffMember(full_name="Ronit Manager", role="manager"))
        noa = staff.get(seed.counselor_ids[0])
        trainee = TraineeRepository(session).add(make_trainee(seed))
        goal = GoalRepository(session).add(Goal(trainee_id=trainee.id, title="Run 5 km", category="physical"))
        PersonalFileRepository(session).add(
            PersonalFile(trainee_id=trainee.id, intake_answers={"Hobbies": "Guitar"})
        )
    return {"manager": manager, "noa": noa, "trainee": trainee, "goal": goal}


def test_update_progress_writes_goal_and_history(engine, academy):
    noa, goal = academy["noa"], academy["goal"]

    with unit_of_work(engine) as session:
        goal_service(session).update_progress(noa, goal.id, 60, "Ran 3 km")
    with unit_of_work(engine) as session:
        goal_service(session).update_progress(noa, goal.id, 100, "Ran 5 km!")

    with unit_of_work(engine) as session:
        loaded = GoalRepository(session).get(goal.id)
        history = goal_service(session).history(noa, goal.id)

    assert (loaded.progress, loaded.status) == (100, GoalStatus.ACHIEVED)
    assert [(u.progress, u.note) for u in history] == [(60, "Ran 3 km"), (100, "Ran 5 km!")]


def test_approval_request_round_trip(engine, academy):
    noa, manager, trainee = academy["noa"], academy["manager"], academy["trainee"]

    with unit_of_work(engine) as session:
        request = approval_service(session).request(
            noa, ApprovalKind.STATUS_CHANGE, trainee.id, "Sent home for 3 days", requested_status="on_leave"
        )
    with unit_of_work(engine) as session:
        approval_service(session).approve(manager, request.id, note="Agreed")

    with unit_of_work(engine) as session:
        loaded = ApprovalRequestRepository(session).get(request.id)
        trainee_now = TraineeRepository(session).get(trainee.id)

    assert loaded.status is ApprovalStatus.APPROVED
    assert loaded.requested_status is TraineeStatus.ON_LEAVE
    assert loaded.decided_by_id == manager.id
    assert loaded.decided_at.utcoffset() is not None
    assert trainee_now.status is TraineeStatus.ON_LEAVE


def test_approval_and_status_change_roll_back_together(engine, academy):
    """If the unit of work fails after approve(), neither the approval nor the new status is saved."""
    noa, manager, trainee = academy["noa"], academy["manager"], academy["trainee"]
    with unit_of_work(engine) as session:
        request = approval_service(session).request(
            noa, ApprovalKind.STATUS_CHANGE, trainee.id, "Wants to leave", requested_status="left"
        )

    with pytest.raises(RuntimeError):
        with unit_of_work(engine) as session:
            approval_service(session).approve(manager, request.id)
            raise RuntimeError("e.g. the notification step failed")

    with unit_of_work(engine) as session:
        assert ApprovalRequestRepository(session).get(request.id).status is ApprovalStatus.PENDING
        assert TraineeRepository(session).get(trainee.id).status is TraineeStatus.ACTIVE


def test_personal_file_needs_approval_in_the_database_too(engine, academy):
    noa, manager, trainee = academy["noa"], academy["manager"], academy["trainee"]
    with unit_of_work(engine) as session:
        request = approval_service(session).request(noa, ApprovalKind.VIEW_PERSONAL_FILE, trainee.id, "Talk")
        pending = ApprovalRequestRepository(session).list_pending()
    assert [r.id for r in pending] == [request.id]

    with unit_of_work(engine) as session:
        approval_service(session).approve(manager, request.id)
    with unit_of_work(engine) as session:
        personal_file = trainee_service(session).get_personal_file(noa, trainee.id)
        still_pending = ApprovalRequestRepository(session).list_pending()

    assert personal_file.intake_answers == {"Hobbies": "Guitar"}
    assert still_pending == []
