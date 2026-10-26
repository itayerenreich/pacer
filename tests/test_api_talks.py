"""The talk endpoints over HTTP, with fakes: the whole counselor flow, and who may do what."""

from collections.abc import Iterator

import pytest
from fake_llm import FakeLLMClient, fake_router
from fakes import FakeUnitOfWork, empty_repos
from fastapi.testclient import TestClient
from test_talk_pipeline import reply

from pacer.api.app import create_app
from pacer.api.dependencies import (
    get_ai_router,
    get_db_engine,
    get_goal_repository,
    get_goal_update_repository,
    get_group_repository,
    get_note_repository,
    get_staff_repository,
    get_talk_proposal_repository,
    get_talk_repository,
    get_trainee_repository,
    get_unit_of_work,
)
from pacer.demo import build_academy
from pacer.demo.talks import MAYA

ITAI, SHIRA, NOA, MANAGER = ({"X-Staff-Id": str(i)} for i in (5, 4, 2, 1))  # Itai: Maya's counselor
TALK = {"held_on": MAYA.held_on.isoformat(), "notes": MAYA.notes}


@pytest.fixture
def repos():
    repos = empty_repos()
    build_academy(repos, schedule=False)
    return repos


@pytest.fixture
def cloud() -> FakeLLMClient:
    return FakeLLMClient(answers=[reply()], runs_locally=False)


@pytest.fixture
def client(repos, cloud) -> Iterator[TestClient]:
    def no_database():
        raise AssertionError("These tests must not use a database")

    app = create_app()
    app.dependency_overrides = {
        get_db_engine: no_database,
        get_unit_of_work: lambda: FakeUnitOfWork(repos),
        get_ai_router: lambda: fake_router(cloud),
        get_staff_repository: lambda: repos.staff,
        get_trainee_repository: lambda: repos.trainees,
        get_group_repository: lambda: repos.groups,
        get_goal_repository: lambda: repos.goals,
        get_goal_update_repository: lambda: repos.goal_updates,
        get_talk_repository: lambda: repos.talks,
        get_talk_proposal_repository: lambda: repos.talk_proposals,
        get_note_repository: lambda: repos.notes,
    }
    with TestClient(app) as test_client:
        yield test_client


def record(client, headers=ITAI) -> int:
    response = client.post(f"/trainees/{MAYA.trainee_id}/talks", json=TALK, headers=headers)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_the_counselor_flow_from_notes_to_goal_progress_and_memory(client, repos):
    talk_id = record(client)
    result = client.post(f"/talks/{talk_id}/summarize", headers=ITAI).json()
    progress, worked, did_not_work, follow_up = result["proposals"]
    assert result["issues"] == [] and result["prompt"] == "talk_summary@v2"

    edited = client.post(f"/talk-proposals/{progress['id']}/approve", json={"progress": 55}, headers=ITAI)
    client.post(f"/talk-proposals/{worked['id']}/approve", headers=ITAI)
    client.post(f"/talk-proposals/{did_not_work['id']}/reject", headers=ITAI)
    client.post(f"/talk-proposals/{follow_up['id']}/approve", json={"note_kind": "follow_up"}, headers=ITAI)

    assert edited.status_code == 200 and edited.json()["text"] is None
    assert client.get("/goals/63/history", headers=ITAI).json()[-1]["progress"] == 55
    notes = client.get(f"/trainees/{MAYA.trainee_id}/notes", headers=SHIRA).json()  # group counselor reads
    assert [n["kind"] for n in notes] == ["worked", "follow_up"]
    assert notes[0]["text"] == "יעד קטן ומדיד עוזר למאיה גולן להתמיד"
    assert client.get(f"/talks/{talk_id}", headers=ITAI).json()["status"] == "reviewed"


def test_who_may_do_what(client):
    talk_id = record(client)
    assert client.post(f"/trainees/{MAYA.trainee_id}/talks", json=TALK, headers=SHIRA).status_code == 403
    assert client.get(f"/talks/{talk_id}", headers=SHIRA).status_code == 200  # same group: may read
    assert client.get(f"/talks/{talk_id}", headers=NOA).status_code == 404  # another group: as if it did not exist
    assert client.post(f"/talks/{talk_id}/summarize", headers=SHIRA).status_code == 403
    proposals = client.post(f"/talks/{talk_id}/summarize", headers=MANAGER).json()["proposals"]
    assert client.post(f"/talk-proposals/{proposals[0]['id']}/approve", headers=SHIRA).status_code == 403


def test_summarizing_twice_is_409_and_a_missing_talk_is_404(client):
    talk_id = record(client)
    client.post(f"/talks/{talk_id}/summarize", headers=ITAI)
    assert client.post(f"/talks/{talk_id}/summarize", headers=ITAI).status_code == 409
    assert client.post("/talks/999/summarize", headers=ITAI).status_code == 404


def test_editing_a_field_of_the_other_kind_is_rejected(client):
    proposals = client.post(f"/talks/{record(client)}/summarize", headers=ITAI).json()["proposals"]
    note_id = proposals[1]["id"]
    assert client.post(f"/talk-proposals/{note_id}/approve", json={"progress": 50}, headers=ITAI).status_code == 409
    assert client.post(f"/talk-proposals/{note_id}/approve", json={"status": "approved"}, headers=ITAI).status_code == 422


def test_notes_by_hand_and_deleting_them(client):
    note = client.post(
        f"/trainees/{MAYA.trainee_id}/notes", json={"kind": "observation", "text": "הובילה את הדיון בקבוצה"}, headers=ITAI
    )
    assert note.status_code == 201
    assert client.delete(f"/notes/{note.json()['id']}", headers=SHIRA).status_code == 403
    assert client.delete(f"/notes/{note.json()['id']}", headers=ITAI).status_code == 204
    assert client.get(f"/trainees/{MAYA.trainee_id}/notes", headers=ITAI).json() == []


def test_a_note_with_an_unknown_kind_or_too_long_is_422(client):
    url = f"/trainees/{MAYA.trainee_id}/notes"
    assert client.post(url, json={"kind": "diagnosis", "text": "משהו ארוך מספיק"}, headers=ITAI).status_code == 422
    assert client.post(url, json={"kind": "observation", "text": "א" * 301}, headers=ITAI).status_code == 422
