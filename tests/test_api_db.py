"""End-to-end: HTTP request -> endpoint -> service -> repository -> PostgreSQL.

Only the engine is overridden (to the test container's database). Everything else
is the production chain from api/dependencies.py, including the unit of work.
"""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Engine, event
from sqlalchemy.orm import Session
from test_repositories import make_trainee, seed_group

from pacer.api.app import create_app
from pacer.api.dependencies import get_db_engine
from pacer.domain import Goal, StaffMember, TraineeStatus
from pacer.repositories import (
    GoalRepository,
    GoalUpdateRepository,
    StaffRepository,
    TraineeRepository,
    unit_of_work,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def academy(engine: Engine) -> dict:
    seed = seed_group(engine)
    with unit_of_work(engine) as session:
        manager = StaffRepository(session).add(StaffMember(full_name="Ronit Manager", role="manager"))
        trainee = TraineeRepository(session).add(make_trainee(seed))
        goal = GoalRepository(session).add(Goal(trainee_id=trainee.id, title="Run 5 km", category="physical"))
    return {"manager_id": manager.id, "noa_id": seed.counselor_ids[0], "trainee": trainee, "goal": goal}


@pytest.fixture
def client(engine: Engine) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_db_engine] = lambda: engine
    # raise_server_exceptions=False: a server error becomes a 500 response, like for a real client
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_progress_update_is_saved_in_the_database(client, engine, academy):
    goal_id, noa = academy["goal"].id, {"X-Staff-Id": str(academy["noa_id"])}

    response = client.post(
        f"/goals/{goal_id}/progress", json={"progress": 60, "note": "Ran 3 km"}, headers=noa
    )
    assert response.status_code == 200

    with unit_of_work(engine) as session:  # a NEW session: reads what was really committed
        assert GoalRepository(session).get(goal_id).progress == 60
        assert [u.note for u in GoalUpdateRepository(session).list_by_goal(goal_id)] == ["Ran 3 km"]


def test_status_change_approval_end_to_end(client, engine, academy):
    trainee_id = academy["trainee"].id
    noa, manager = {"X-Staff-Id": str(academy["noa_id"])}, {"X-Staff-Id": str(academy["manager_id"])}

    created = client.post(
        "/approvals",
        json={
            "kind": "status_change",
            "trainee_id": trainee_id,
            "reason": "Wants to leave",
            "requested_status": "left",
        },
        headers=noa,
    )
    approved = client.post(
        f"/approvals/{created.json()['id']}/approve", json={"note": "After a talk"}, headers=manager
    )

    assert (created.status_code, approved.status_code) == (201, 200)
    with unit_of_work(engine) as session:
        assert TraineeRepository(session).get(trainee_id).status is TraineeStatus.LEFT


def test_a_failed_commit_is_never_reported_as_success(client, engine, academy):
    """The unit of work commits BEFORE the response is sent (scope="function" in dependencies.py).

    If the commit fails, the client must get an error, and nothing may be saved. With
    FastAPI's default scope, the client would already have received "200 OK".
    """
    goal_id, noa = academy["goal"].id, {"X-Staff-Id": str(academy["noa_id"])}

    def fail_commit(session):
        raise RuntimeError("the database rejected the commit")

    event.listen(Session, "before_commit", fail_commit)
    try:
        response = client.post(f"/goals/{goal_id}/progress", json={"progress": 60, "note": "x"}, headers=noa)
    finally:
        event.remove(Session, "before_commit", fail_commit)

    assert response.status_code == 500
    with unit_of_work(engine) as session:
        assert GoalRepository(session).get(goal_id).progress == 0
        assert GoalUpdateRepository(session).list_by_goal(goal_id) == []
