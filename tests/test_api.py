"""API tests over HTTP, with no database: Dependency Injection at work.

app.dependency_overrides swaps each repository dependency for an in-memory fake
(fakes.py). The real endpoints, real services and real permission rules run;
only the storage underneath is different. The engine dependency is replaced by
one that fails, which proves these tests never touch a database.
"""

from collections.abc import Iterator

import pytest
from fakes import World
from fastapi.testclient import TestClient

from pacer.api.app import create_app
from pacer.api.dependencies import (
    get_approval_repository,
    get_db_engine,
    get_goal_repository,
    get_goal_update_repository,
    get_group_repository,
    get_personal_file_repository,
    get_staff_repository,
    get_trainee_repository,
)
from pacer.domain import GoalStatus, StaffMember, TraineeStatus


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def client(world: World) -> Iterator[TestClient]:
    def no_database():
        raise AssertionError("These tests must not use a database")

    app = create_app()
    app.dependency_overrides = {
        get_db_engine: no_database,
        get_staff_repository: lambda: world.staff,
        get_group_repository: lambda: world.groups,
        get_trainee_repository: lambda: world.trainees,
        get_personal_file_repository: lambda: world.personal_files,
        get_goal_repository: lambda: world.goals,
        get_goal_update_repository: lambda: world.goal_updates,
        get_approval_repository: lambda: world.approvals,
    }
    with TestClient(app) as test_client:
        yield test_client


def as_(staff: StaffMember) -> dict[str, str]:
    """Request headers that say who is acting."""
    return {"X-Staff-Id": str(staff.id)}


# --- who are you (401) ----------------------------------------------------------------


def test_me_returns_the_acting_staff_member(client, world):
    response = client.get("/me", headers=as_(world.noa))
    assert response.status_code == 200
    assert response.json()["full_name"] == "Noa Levi"


def test_missing_header_is_401(client):
    response = client.get("/me")
    assert response.status_code == 401
    assert response.json() == {"detail": "Missing X-Staff-Id header"}


def test_unknown_staff_id_is_401(client):
    assert client.get("/me", headers={"X-Staff-Id": "999"}).status_code == 401


# --- goals --------------------------------------------------------------------------


def test_update_progress_and_read_history(client, world):
    url = f"/goals/{world.goal.id}/progress"
    first = client.post(url, json={"progress": 40, "note": "Ran 2 km"}, headers=as_(world.noa))
    second = client.post(url, json={"progress": 100, "note": "Ran 5 km!"}, headers=as_(world.noa))

    assert first.status_code == 200
    assert first.json()["status"] == "in_progress"
    assert second.json()["status"] == "achieved"
    history = client.get(f"/goals/{world.goal.id}/history", headers=as_(world.omer)).json()
    assert [(h["progress"], h["note"]) for h in history] == [(40, "Ran 2 km"), (100, "Ran 5 km!")]


def test_co_counselor_cannot_update_progress_403(client, world):
    response = client.post(
        f"/goals/{world.goal.id}/progress", json={"progress": 40, "note": "x"}, headers=as_(world.omer)
    )
    assert response.status_code == 403
    assert "Not allowed" in response.json()["detail"]


def test_missing_goal_is_404(client, world):
    response = client.post("/goals/999/progress", json={"progress": 40, "note": "x"}, headers=as_(world.noa))
    assert response.status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"progress": 150, "note": "Too much"},  # out of range
        {"progress": 40},  # note missing
        {"progress": 40, "note": ""},  # empty note
        {"progress": 40, "note": "ok", "author_id": 1},  # unknown field: the author is the actor
    ],
)
def test_invalid_progress_body_is_422(client, world, body):
    response = client.post(f"/goals/{world.goal.id}/progress", json=body, headers=as_(world.noa))
    assert response.status_code == 422
    assert world.goal_updates.list_by_goal(world.goal.id) == []  # nothing was saved


def test_dropped_goal_is_409(client, world):
    world.goals.update(world.goal.model_copy(update={"status": GoalStatus.DROPPED}))
    response = client.post(
        f"/goals/{world.goal.id}/progress", json={"progress": 40, "note": "x"}, headers=as_(world.noa)
    )
    assert response.status_code == 409


def test_add_and_list_goals(client, world):
    response = client.post(
        f"/trainees/{world.yael.id}/goals",
        json={"title": "Lead a discussion", "category": "leadership", "target_date": "2027-01-15"},
        headers=as_(world.noa),
    )
    assert response.status_code == 201
    assert response.json()["progress"] == 0
    goals = client.get(f"/trainees/{world.yael.id}/goals", headers=as_(world.omer)).json()
    assert [g["title"] for g in goals] == ["Run 5 km", "Lead a discussion"]


def test_goal_title_rule_comes_from_the_domain(client, world):
    response = client.post(
        f"/trainees/{world.yael.id}/goals", json={"title": "ab", "category": "social"}, headers=as_(world.noa)
    )
    assert response.status_code == 422


def test_no_goals_for_a_trainee_who_left_409(client, world):
    world.trainees.update(world.yael.with_status(TraineeStatus.LEFT))
    response = client.post(
        f"/trainees/{world.yael.id}/goals",
        json={"title": "Anything", "category": "social"},
        headers=as_(world.noa),
    )
    assert response.status_code == 409


# --- trainees -----------------------------------------------------------------------


def new_trainee_body(world: World, **overrides) -> dict:
    body = {
        "first_name": "Itamar",
        "last_name": "Peretz",
        "cohort_id": world.cohort.id,
        "group_id": world.group_a.id,
        "primary_counselor_id": world.omer.id,
    }
    body.update(overrides)
    return body


def test_manager_enrolls_a_trainee_201(client, world):
    response = client.post("/trainees", json=new_trainee_body(world), headers=as_(world.manager))
    assert response.status_code == 201
    assert response.json()["status"] == "active"
    assert client.get(f"/trainees/{response.json()['id']}", headers=as_(world.omer)).status_code == 200


def test_counselor_cannot_enroll_403(client, world):
    assert client.post("/trainees", json=new_trainee_body(world), headers=as_(world.noa)).status_code == 403


def test_enroll_with_counselor_outside_the_group_409(client, world):
    body = new_trainee_body(world, primary_counselor_id=world.dana.id)
    assert client.post("/trainees", json=body, headers=as_(world.manager)).status_code == 409


def test_counselor_of_another_group_gets_404_as_if_the_trainee_did_not_exist(client, world):
    hidden = client.get(f"/trainees/{world.yael.id}", headers=as_(world.dana))
    missing = client.get("/trainees/999", headers=as_(world.dana))
    assert hidden.status_code == missing.status_code == 404
    assert hidden.json()["detail"] == f"Trainee {world.yael.id} does not exist"  # same words: nothing to learn


def test_manager_changes_status(client, world):
    response = client.put(
        f"/trainees/{world.yael.id}/status", json={"status": "on_leave"}, headers=as_(world.manager)
    )
    assert response.status_code == 200
    assert world.trainees.get(world.yael.id).status is TraineeStatus.ON_LEAVE


def test_unknown_status_value_is_422(client, world):
    response = client.put(
        f"/trainees/{world.yael.id}/status", json={"status": "expelled"}, headers=as_(world.manager)
    )
    assert response.status_code == 422


# --- approvals: the full flows over HTTP ------------------------------------------------


def test_personal_file_flow(client, world):
    file_url = f"/trainees/{world.yael.id}/personal-file"
    assert client.get(file_url, headers=as_(world.noa)).status_code == 403

    created = client.post(
        "/approvals",
        json={"kind": "view_personal_file", "trainee_id": world.yael.id, "reason": "Personal talk tomorrow"},
        headers=as_(world.noa),
    )
    assert created.status_code == 201
    request_id = created.json()["id"]
    assert [r["id"] for r in client.get("/approvals/pending", headers=as_(world.manager)).json()] == [
        request_id
    ]

    approved = client.post(
        f"/approvals/{request_id}/approve", json={"note": "OK"}, headers=as_(world.manager)
    )
    assert approved.json()["status"] == "approved"

    response = client.get(file_url, headers=as_(world.noa))
    assert response.status_code == 200
    assert response.json()["intake_answers"] == {"Hobbies": "Guitar"}


def test_status_change_flow_without_a_note(client, world):
    created = client.post(
        "/approvals",
        json={
            "kind": "status_change",
            "trainee_id": world.yael.id,
            "reason": "Sent home for 3 days",
            "requested_status": "on_leave",
        },
        headers=as_(world.noa),
    )
    response = client.post(
        f"/approvals/{created.json()['id']}/approve", headers=as_(world.manager)
    )  # no body

    assert response.status_code == 200
    assert world.trainees.get(world.yael.id).status is TraineeStatus.ON_LEAVE


def test_status_change_without_requested_status_is_422(client, world):
    response = client.post(
        "/approvals",
        json={"kind": "status_change", "trainee_id": world.yael.id, "reason": "Leaving"},
        headers=as_(world.noa),
    )
    assert response.status_code == 422  # the domain rule, through the ValidationError handler


def test_deciding_twice_is_409(client, world):
    created = client.post(
        "/approvals",
        json={"kind": "view_personal_file", "trainee_id": world.yael.id, "reason": "Talk"},
        headers=as_(world.noa),
    )
    request_id = created.json()["id"]
    client.post(f"/approvals/{request_id}/reject", headers=as_(world.manager))

    response = client.post(f"/approvals/{request_id}/approve", headers=as_(world.manager))
    assert response.status_code == 409  # InvalidTransitionError, a subclass of DomainRuleError
    assert "rejected" in response.json()["detail"]


def test_counselor_cannot_approve_403(client, world):
    created = client.post(
        "/approvals",
        json={"kind": "view_personal_file", "trainee_id": world.yael.id, "reason": "Talk"},
        headers=as_(world.noa),
    )
    assert (
        client.post(f"/approvals/{created.json()['id']}/approve", headers=as_(world.omer)).status_code == 403
    )


# --- documentation ------------------------------------------------------------------


def test_docs_page_and_schema_are_served(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert "/goals/{goal_id}/progress" in paths
    assert paths["/goals/{goal_id}/progress"]["post"]["requestBody"]
