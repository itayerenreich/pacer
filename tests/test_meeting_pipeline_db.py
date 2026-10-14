"""The meeting pipeline against PostgreSQL: real transactions, real row locks, real threads.

The fake tests prove the logic. These prove what only a database can show:
* while the model works, our connection is NOT "idle in transaction" (no transaction held open);
* two requests processing the SAME meeting at once: one wins, the other's answer is discarded,
  and nothing is saved twice (SELECT ... FOR UPDATE + checking the status again);
* two requests processing DIFFERENT meetings at once: both succeed, each with its own proposals;
* the summary and the proposals are committed together, or not at all.
"""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fake_llm import FakeLLMClient, fake_router
from fastapi.testclient import TestClient
from sqlalchemy import text
from test_meeting_pipeline import WEEKLY_ANSWER

from pacer.ai.meeting_pipeline import MeetingPipeline
from pacer.api.app import create_app
from pacer.api.dependencies import get_ai_router, get_db_engine
from pacer.demo import build_academy
from pacer.demo.meetings import SHORT, WEEKLY, SampleMeeting
from pacer.domain import MeetingStatus
from pacer.repositories import repositories
from pacer.services import MeetingService, RuleViolationError

pytestmark = pytest.mark.integration
GOOD = json.dumps(WEEKLY_ANSWER, ensure_ascii=False)


@pytest.fixture
def academy(engine):
    with repositories(engine) as repos:
        build_academy(repos)
    return engine


def uow_for(engine):
    return lambda: repositories(engine)


def manager(engine):
    with repositories(engine) as repos:
        return repos.staff.get(1)


def record(engine, sample: SampleMeeting = WEEKLY) -> int:
    with repositories(engine) as repos:
        service = MeetingService(repos.meetings, repos.proposals, repos.tasks, repos.staff)
        return service.create_meeting(
            manager(engine),
            title=sample.title,
            held_on=sample.held_on,
            participant_ids=sample.participant_ids,
            transcript=sample.transcript,
        ).id


def idle_in_transaction(engine) -> int:
    """Connections to the test database that sit inside an open transaction, doing nothing."""
    with engine.connect() as connection:
        return connection.execute(
            text(
                "SELECT count(*) FROM pg_stat_activity "
                "WHERE datname = current_database() AND state LIKE 'idle in transaction%'"
            )
        ).scalar()


def test_over_http_from_transcript_to_tasks_in_postgresql(academy):
    seen: list[int] = []

    def model(message: str) -> str:  # the endpoint itself must not hold the request's transaction open
        seen.append(idle_in_transaction(academy))
        return GOOD

    cloud = FakeLLMClient(answers=model, runs_locally=False)
    app = create_app()
    app.dependency_overrides = {get_db_engine: lambda: academy, get_ai_router: lambda: fake_router(cloud)}
    body = {
        "title": WEEKLY.title,
        "held_on": WEEKLY.held_on.isoformat(),
        "participant_ids": WEEKLY.participant_ids,
        "transcript": WEEKLY.transcript,
    }
    with TestClient(app) as client:
        meeting_id = client.post("/meetings", json=body, headers={"X-Staff-Id": "1"}).json()["id"]
        result = client.post(f"/meetings/{meeting_id}/process", headers={"X-Staff-Id": "1"}).json()
        talk = result["proposals"][3]
        client.post(f"/proposals/{talk['id']}/approve", headers={"X-Staff-Id": "1"})
        omer_tasks = client.get("/tasks/mine", headers={"X-Staff-Id": "3"}).json()
        stored = client.get(f"/meetings/{meeting_id}", headers={"X-Staff-Id": "1"}).json()

    assert seen == [0]
    assert [t["description"] for t in omer_tasks] == ["לשוחח עם אדל אזולאי על ריצות הבוקר"]
    assert "[TRAINEE_16]" in stored["anonymized_transcript"] and "אדל" not in stored["anonymized_transcript"]
    assert "אדל אזולאי" in stored["summary"]  # names restored on this computer, then saved


def test_no_transaction_is_held_open_while_the_model_works(academy):
    meeting_id = record(academy)
    seen: list[int] = []

    def model(message: str) -> str:
        seen.append(idle_in_transaction(academy))
        return GOOD

    cloud = FakeLLMClient(answers=model, runs_locally=False)
    MeetingPipeline(uow_for(academy), fake_router(cloud)).process(manager(academy), meeting_id)
    assert seen == [0]


def test_the_same_meeting_twice_at_once_one_wins_and_nothing_is_saved_twice(academy):
    meeting_id = record(academy)
    both_in_the_model = threading.Barrier(2, timeout=10)

    def model(message: str) -> str:
        both_in_the_model.wait()  # both requests read ANONYMIZED, and both are now "thinking"
        return GOOD

    def run():
        cloud = FakeLLMClient(answers=model, runs_locally=False)
        return MeetingPipeline(uow_for(academy), fake_router(cloud)).extract(manager(academy), meeting_id)

    MeetingPipeline(uow_for(academy), fake_router(FakeLLMClient(runs_locally=False))).anonymize(
        manager(academy), meeting_id
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run), pool.submit(run)]
    outcomes = [f.exception() or f.result() for f in futures]

    errors = [o for o in outcomes if isinstance(o, Exception)]
    assert len(errors) == 1 and isinstance(errors[0], RuleViolationError)
    assert "another request" in str(errors[0])
    with repositories(academy) as repos:
        assert len(repos.proposals.list_by_meeting(meeting_id)) == 4  # one answer's worth, not eight
        assert repos.meetings.get(meeting_id).status is MeetingStatus.PENDING_APPROVAL


def test_two_meetings_at_once_are_processed_in_parallel_and_kept_apart(academy):
    weekly, short = record(academy, WEEKLY), record(academy, SHORT)
    both_in_the_model = threading.Barrier(2, timeout=10)  # proves they really overlap
    short_answer = json.dumps(
        {
            "summary": "ישיבת לוגיסטיקה קצרה על הציוד לטיול ועל בקשת יציאה הביתה.",
            "tasks": [{"description": "לבדוק את בקשת היציאה של [NAME_1]", "owner": "[STAFF_3]", "due_date": "2026-10-14"}],
        },
        ensure_ascii=False,
    )

    def model(message: str) -> str:
        both_in_the_model.wait()
        return short_answer if "[NAME_1]" in message else GOOD

    def run(meeting_id: int):
        cloud = FakeLLMClient(answers=model, runs_locally=False)
        return MeetingPipeline(uow_for(academy), fake_router(cloud)).process(manager(academy), meeting_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        weekly_result, short_result = pool.map(run, [weekly, short])

    assert len(weekly_result.proposals) == 4 and {p.meeting_id for p in weekly_result.proposals} == {weekly}
    assert [p.description for p in short_result.proposals] == ["לבדוק את בקשת היציאה של זיו"]
    assert {p.meeting_id for p in short_result.proposals} == {short}


def test_a_failed_write_leaves_no_summary_and_no_proposals(academy, monkeypatch):
    def broken(self, meeting_id, drafts):
        raise RuntimeError("the write failed half way")

    monkeypatch.setattr(MeetingService, "submit_proposals", broken)
    meeting_id = record(academy)
    with pytest.raises(RuntimeError):
        MeetingPipeline(uow_for(academy), fake_router(FakeLLMClient(answers=[GOOD], runs_locally=False))).process(
            manager(academy), meeting_id
        )
    with repositories(academy) as repos:
        meeting = repos.meetings.get(meeting_id)
        assert meeting.status is MeetingStatus.ANONYMIZED and meeting.summary is None
        assert repos.proposals.list_by_meeting(meeting_id) == []
