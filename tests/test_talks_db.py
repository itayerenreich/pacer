"""Talks, talk proposals and notes against PostgreSQL: the fourth migration, and the counselor flow."""

import pytest
from fake_llm import FakeLLMClient, fake_router
from fastapi.testclient import TestClient
from test_meeting_pipeline_db import idle_in_transaction
from test_talk_pipeline import reply

from pacer.api.app import create_app
from pacer.api.dependencies import get_ai_router, get_db_engine
from pacer.demo import build_academy
from pacer.demo.talks import MAYA
from pacer.domain import NoteKind, ProposalStatus, TalkProposalKind, TalkStatus
from pacer.repositories import repositories

pytestmark = pytest.mark.integration
ITAI = {"X-Staff-Id": "5"}


@pytest.fixture
def academy(engine):
    with repositories(engine) as repos:
        build_academy(repos)
    return engine


def test_the_whole_flow_over_http_in_postgresql(academy):
    seen: list[int] = []

    def model(message: str) -> str:
        seen.append(idle_in_transaction(academy))
        return reply()

    cloud = FakeLLMClient(answers=model, runs_locally=False)
    app = create_app()
    app.dependency_overrides = {get_db_engine: lambda: academy, get_ai_router: lambda: fake_router(cloud)}
    body = {"held_on": MAYA.held_on.isoformat(), "notes": MAYA.notes}
    with TestClient(app) as client:
        talk_id = client.post(f"/trainees/{MAYA.trainee_id}/talks", json=body, headers=ITAI).json()["id"]
        progress, worked, *_ = client.post(f"/talks/{talk_id}/summarize", headers=ITAI).json()["proposals"]
        client.post(f"/talk-proposals/{progress['id']}/approve", headers=ITAI)
        client.post(f"/talk-proposals/{worked['id']}/approve", headers=ITAI)
        talk = client.get(f"/talks/{talk_id}", headers=ITAI).json()

    assert seen == [0]  # no transaction held open while the model worked
    assert "[TRAINEE_22]" in talk["anonymized_notes"] and "מאיה" not in talk["anonymized_notes"]
    assert "מאיה גולן" in talk["summary"]
    with repositories(academy) as repos:
        assert repos.goals.get(63).progress == 60
        [note] = repos.notes.list_by_trainee(MAYA.trainee_id)
        assert (note.kind, note.source_talk_id, note.author_id) == (NoteKind.WORKED, talk_id, 5)


def test_deleting_a_note_leaves_a_valid_proposal_pointing_nowhere(academy):
    """ON DELETE SET NULL in the database, and the domain accepts what the database leaves."""
    from pacer.ai.talk_pipeline import talk_service
    from pacer.services import TalkDraft

    with repositories(academy) as repos:
        itai = repos.staff.get(5)
        service = talk_service(repos)
        talk = service.record_talk(itai, MAYA.trainee_id, held_on=MAYA.held_on, notes=MAYA.notes)
        repos.talks.update(talk.advance(TalkStatus.SUMMARIZED, summary="סיכום", anonymized_notes="טקסט"))
        draft = TalkDraft(kind=TalkProposalKind.NOTE, note_kind=NoteKind.WORKED, text="יעד קטן עוזר לה")
        [proposal] = service.submit_proposals(talk.id, [draft])
        note_id = service.approve_proposal(itai, proposal.id).note_id
    with repositories(academy) as repos:
        talk_service(repos).delete_note(repos.staff.get(5), note_id)
    with repositories(academy) as repos:
        stored = repos.talk_proposals.get(proposal.id)
        assert repos.notes.get(note_id) is None
        assert (stored.status, stored.note_id, stored.text) == (ProposalStatus.APPROVED, None, None)


def test_hebrew_notes_round_trip(academy):
    from pacer.ai.talk_pipeline import talk_service

    with repositories(academy) as repos:
        talk = talk_service(repos).record_talk(repos.staff.get(5), MAYA.trainee_id, held_on=MAYA.held_on, notes=MAYA.notes)
    with repositories(academy) as repos:
        assert repos.talks.get(talk.id).notes == MAYA.notes
        assert repos.talks.list_by_trainee(MAYA.trainee_id)[0].id == talk.id
