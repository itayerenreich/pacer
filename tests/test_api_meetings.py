"""Meeting and task endpoints over HTTP with fakes: status codes and permissions."""

from collections.abc import Iterator

import pytest
from fakes import World
from fastapi.testclient import TestClient

from pacer.api.app import create_app
from pacer.api.dependencies import (
    get_db_engine,
    get_meeting_repository,
    get_proposal_repository,
    get_staff_repository,
    get_task_repository,
)
from pacer.domain import MeetingStatus, StaffMember
from pacer.services import MeetingService, TaskDraft

BODY = {
    "title": "ישיבת צוות",
    "held_on": "2026-10-12",
    "participant_ids": [1, 2, 3],
    "transcript": "נועה: נתחיל. עומר: צריך לתאם הסעה לטיול.",
}


@pytest.fixture
def world() -> World:
    return World()


@pytest.fixture
def client(world) -> Iterator[TestClient]:
    def no_database():
        raise AssertionError("These tests must not use a database")

    app = create_app()
    app.dependency_overrides = {
        get_db_engine: no_database,
        get_staff_repository: lambda: world.staff,
        get_meeting_repository: lambda: world.meetings,
        get_proposal_repository: lambda: world.proposals,
        get_task_repository: lambda: world.tasks,
    }
    with TestClient(app) as test_client:
        yield test_client


def as_(staff: StaffMember) -> dict[str, str]:
    return {"X-Staff-Id": str(staff.id)}


def with_proposals(world: World, client: TestClient) -> list:
    meeting_id = client.post("/meetings", json=BODY, headers=as_(world.manager)).json()["id"]
    meeting = world.meetings.get(meeting_id).advance(MeetingStatus.ANONYMIZED, anonymized_transcript="טקסט")
    world.meetings.update(meeting.advance(MeetingStatus.EXTRACTED, summary="סיכום"))
    service = MeetingService(world.meetings, world.proposals, world.tasks, world.staff)
    return service.submit_proposals(
        meeting_id,
        [
            TaskDraft(description="לתאם הסעה לטיול", owner_id=world.omer.id),
            TaskDraft(description="להזמין ציוד"),
        ],
    )


def test_manager_records_a_meeting_201(client, world):
    response = client.post("/meetings", json=BODY, headers=as_(world.manager))
    assert response.status_code == 201
    assert response.json()["status"] == "transcribed"


def test_counselors_cannot_record_or_read_meetings_403(client, world):
    assert client.post("/meetings", json=BODY, headers=as_(world.noa)).status_code == 403
    meeting_id = client.post("/meetings", json=BODY, headers=as_(world.manager)).json()["id"]
    assert client.get(f"/meetings/{meeting_id}", headers=as_(world.noa)).status_code == 403
    assert client.get(f"/meetings/{meeting_id}/proposals", headers=as_(world.noa)).status_code == 403


def test_a_too_short_transcript_is_422(client, world):
    response = client.post("/meetings", json={**BODY, "transcript": "קצר"}, headers=as_(world.manager))
    assert response.status_code == 422


def test_approve_with_edits_and_reject(client, world):
    transport, equipment = with_proposals(world, client)

    approved = client.post(
        f"/proposals/{transport.id}/approve", json={"due_date": "2026-10-20"}, headers=as_(world.manager)
    )
    no_owner = client.post(f"/proposals/{equipment.id}/approve", headers=as_(world.manager))
    rejected = client.post(f"/proposals/{equipment.id}/reject", headers=as_(world.manager))

    assert approved.status_code == 200 and approved.json()["due_date"] == "2026-10-20"
    assert no_owner.status_code == 409  # the model named no one: choose an owner first
    assert rejected.json()["status"] == "rejected"


def test_editing_fields_that_are_not_editable_is_422(client, world):
    transport, _ = with_proposals(world, client)
    response = client.post(
        f"/proposals/{transport.id}/approve", json={"status": "approved"}, headers=as_(world.manager)
    )
    assert response.status_code == 422


def test_deciding_twice_is_409(client, world):
    transport, _ = with_proposals(world, client)
    client.post(f"/proposals/{transport.id}/reject", headers=as_(world.manager))
    assert client.post(f"/proposals/{transport.id}/approve", headers=as_(world.manager)).status_code == 409


def test_tasks_mine_and_complete(client, world):
    transport, _ = with_proposals(world, client)
    task_id = client.post(f"/proposals/{transport.id}/approve", headers=as_(world.manager)).json()["id"]

    assert [t["id"] for t in client.get("/tasks/mine", headers=as_(world.omer)).json()] == [task_id]
    assert client.get("/tasks/mine", headers=as_(world.noa)).json() == []
    assert client.post(f"/tasks/{task_id}/complete", headers=as_(world.noa)).status_code == 404  # not hers
    assert client.post(f"/tasks/{task_id}/complete", headers=as_(world.omer)).json()["status"] == "done"
    assert client.get("/tasks", headers=as_(world.omer)).status_code == 403
    assert len(client.get("/tasks", headers=as_(world.manager)).json()) == 1


def test_held_on_must_be_a_date(client, world):
    response = client.post("/meetings", json={**BODY, "held_on": "next week"}, headers=as_(world.manager))
    assert response.status_code == 422
