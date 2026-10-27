"""The pipeline endpoints over HTTP, with fakes: status codes, the result shape, and what is hidden."""

import json
from collections.abc import Iterator

import pytest
from fake_llm import FakeLLMClient, fake_router
from fakes import FakeUnitOfWork, empty_repos
from fastapi.testclient import TestClient
from test_meeting_pipeline import WEEKLY_ANSWER

from pacer.ai import LLMConfigError
from pacer.api.app import create_app
from pacer.api.dependencies import (
    get_ai_router,
    get_db_engine,
    get_group_repository,
    get_meeting_repository,
    get_proposal_repository,
    get_staff_repository,
    get_task_repository,
    get_trainee_repository,
    get_unit_of_work,
)
from pacer.demo import build_academy
from pacer.demo.meetings import WEEKLY

MANAGER, COUNSELOR = {"X-Staff-Id": "1"}, {"X-Staff-Id": "2"}
BODY = {
    "title": WEEKLY.title,
    "held_on": WEEKLY.held_on.isoformat(),
    "participant_ids": WEEKLY.participant_ids,
    "transcript": WEEKLY.transcript,
}


class Setup:
    def __init__(self, tmp_path) -> None:
        self.repos = empty_repos()
        build_academy(self.repos, schedule=False)
        self.uow = FakeUnitOfWork(self.repos)
        self.cloud = FakeLLMClient(answers=[json.dumps(WEEKLY_ANSWER, ensure_ascii=False)], runs_locally=False)


@pytest.fixture
def setup(tmp_path) -> Setup:
    return Setup(tmp_path)


@pytest.fixture
def client(setup) -> Iterator[TestClient]:
    def no_database():
        raise AssertionError("These tests must not use a database")

    app = create_app()
    app.dependency_overrides = {
        get_db_engine: no_database,
        get_unit_of_work: lambda: setup.uow,
        get_ai_router: lambda: fake_router(setup.cloud),
        # the plain endpoints (create, approve) use the request's repositories
        get_staff_repository: lambda: setup.repos.staff,
        get_meeting_repository: lambda: setup.repos.meetings,
        get_proposal_repository: lambda: setup.repos.proposals,
        get_task_repository: lambda: setup.repos.tasks,
        get_trainee_repository: lambda: setup.repos.trainees,
        get_group_repository: lambda: setup.repos.groups,
    }
    with TestClient(app) as test_client:
        yield test_client


def create(client, **changes) -> int:
    response = client.post("/meetings", json={**BODY, **changes}, headers=MANAGER)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_process_returns_proposals_issues_and_the_prompt_version(client):
    response = client.post(f"/meetings/{create(client)}/process", headers=MANAGER)

    assert response.status_code == 200
    body = response.json()
    assert body["meeting"]["status"] == "pending_approval"
    assert [p["owner_id"] for p in body["proposals"]] == [5, 2, 4, 3]
    assert body["proposals"][3]["description"] == "לשוחח עם אדל אזולאי על ריצות הבוקר"
    assert body["issues"] == []
    assert body["prompt"] == "meeting_extract@v3" and body["attempts"] == 1


def test_validation_issues_reach_the_manager(client, setup):
    answer = {**WEEKLY_ANSWER, "tasks": [{**WEEKLY_ANSWER["tasks"][0], "due_date": "2026-01-01"}]}
    setup.cloud.answers = [json.dumps(answer, ensure_ascii=False)]
    body = client.post(f"/meetings/{create(client)}/process", headers=MANAGER).json()
    assert body["issues"] == [
        {
            "code": "due_before_meeting",
            "item": 1,
            "detail": "2026-01-01 is before the meeting; date cleared",
            "section": None,
        }
    ]


def test_the_whole_flow_from_transcript_to_a_counselors_task(client):
    meeting_id = create(client)
    proposals = client.post(f"/meetings/{meeting_id}/process", headers=MANAGER).json()["proposals"]
    transport, volunteering = proposals[0], proposals[1]

    client.post(f"/proposals/{transport['id']}/approve", json={"due_date": "2026-10-16"}, headers=MANAGER)
    client.post(f"/proposals/{volunteering['id']}/reject", headers=MANAGER)

    itai_tasks = client.get("/tasks/mine", headers={"X-Staff-Id": "5"}).json()
    assert [(t["description"], t["due_date"]) for t in itai_tasks] == [
        ("לתאם את ההסעה לטיול השנתי עם חברת האוטובוסים", "2026-10-16")
    ]


def test_a_counselor_cannot_process_and_no_model_is_called(client, setup):
    meeting_id = create(client)
    assert client.post(f"/meetings/{meeting_id}/process", headers=COUNSELOR).status_code == 403
    assert setup.cloud.calls == []


def test_no_identity_is_401_and_an_unknown_meeting_is_404(client):
    assert client.post("/meetings/1/process").status_code == 401
    assert client.post("/meetings/999/process", headers=MANAGER).status_code == 404


def test_processing_twice_is_409(client):
    meeting_id = create(client)
    client.post(f"/meetings/{meeting_id}/process", headers=MANAGER)
    assert client.post(f"/meetings/{meeting_id}/process", headers=MANAGER).status_code == 409


def test_an_invalid_model_answer_is_502_and_nothing_is_saved(client, setup):
    setup.cloud.answers = ["לא JSON בכלל"]
    meeting_id = create(client)
    response = client.post(f"/meetings/{meeting_id}/process", headers=MANAGER)
    assert response.status_code == 502
    assert "not saved" in response.json()["detail"]
    assert client.get(f"/meetings/{meeting_id}", headers=MANAGER).json()["status"] == "anonymized"


def test_a_missing_api_key_is_503_with_a_fixed_message(client, setup):
    def no_key(message):
        raise LLMConfigError("ANTHROPIC_API_KEY is missing (provider text that must not be forwarded)")

    setup.cloud.answers = no_key
    response = client.post(f"/meetings/{create(client)}/process", headers=MANAGER)
    assert response.status_code == 503
    assert "provider text" not in response.json()["detail"]
