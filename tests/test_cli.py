"""The command line tools people run by hand: the connection check, and the two demo runners.

No real model and no network: the clients are fakes. The demo runners run against PostgreSQL
(integration), because that is what they are for.
"""

import json

import pytest
from fake_llm import FakeLLMClient, fake_router

import pacer.ai.check as check
import pacer.demo.meeting as meeting_cli
import pacer.demo.talk as talk_cli
from pacer.ai import LLMUnavailableError
from pacer.ai.anonymization import Anonymizer, load_roster
from pacer.ai.routing import MODELS, ModelSpec, Router
from pacer.demo.__main__ import load
from pacer.repositories import repositories

DIGEST = json.dumps({"language": "he", "summary": "המחזור יצא להתנדבות בצפון.", "topics": ["התנדבות"]})


def unavailable():
    raise LLMUnavailableError("no API key")


# --- python -m pacer.ai.check ---------------------------------------------------------------------


@pytest.fixture
def fakes(monkeypatch):
    cloud = FakeLLMClient(answers=[DIGEST], model="fake/claude", runs_locally=False)
    local = FakeLLMClient(answers=["מכינה היא שנת הכנה לפני הצבא."], model="fake/ollama")
    monkeypatch.setattr(check, "claude_client", lambda: cloud)
    monkeypatch.setattr(check, "ollama_client", lambda: local)
    monkeypatch.setattr(check, "get_router", lambda: fake_router(cloud))
    return cloud, local


def test_the_check_passes_when_every_model_answers(fakes, capsys):
    with pytest.raises(SystemExit) as done:
        check.main([])
    out = capsys.readouterr().out
    assert done.value.code == 0
    assert "connection_check@v1" in out and "Attempts: 1" in out


def test_an_unreachable_local_model_fails_the_check_with_a_hint(fakes, monkeypatch, capsys):
    monkeypatch.setattr(check, "ollama_client", unavailable)
    with pytest.raises(SystemExit) as done:
        check.main(["--only", "local"])
    assert done.value.code == 1
    assert "is Ollama running?" in capsys.readouterr().out


def test_an_invalid_structured_answer_fails_the_check(fakes, capsys):
    cloud, _ = fakes
    cloud.answers = ["not json"]
    with pytest.raises(SystemExit) as done:
        check.main(["--only", "structured"])
    assert done.value.code == 1
    assert "Failed after 2 attempt(s)" in capsys.readouterr().out


# --- python -m pacer.demo.meeting / pacer.demo.talk (PostgreSQL) -------------------------------------


@pytest.fixture
def academy(engine, monkeypatch):
    load(engine, reset=False, seed=2026)
    for cli in (meeting_cli, talk_cli):
        monkeypatch.setattr(cli, "get_engine", lambda: engine)
    return engine


def no_known_name(engine, text: str) -> bool:
    with repositories(engine) as repos:
        return not Anonymizer(load_roster(repos.staff, repos.trainees)).contains_known_name(text)


@pytest.mark.integration
def test_the_meeting_dry_run_shows_only_tokens_and_sends_nothing(academy, tmp_path, capsys):
    out = tmp_path / "weekly.md"
    meeting_cli.main(["weekly", "--out", str(out)])
    report = out.read_text(encoding="utf-8")
    assert "Nothing was sent" in report and "[STAFF_1]" in report
    message = report.split("```")[1]
    assert no_known_name(academy, message)
    assert "Written to" in capsys.readouterr().out


@pytest.mark.integration
def test_the_meeting_send_saves_proposals_for_the_manager(academy, monkeypatch, capsys):
    answer = {"summary": "[STAFF_1] פתחה את הישיבה וסיכמה את ההסעה.", "tasks": [
        {"description": "לתאם את ההסעה לטיול השנתי", "owner": "[STAFF_5]", "due_date": "2026-10-15"}]}
    cloud = FakeLLMClient(answers=[json.dumps(answer, ensure_ascii=False)], runs_locally=False)
    monkeypatch.setattr(meeting_cli, "get_router", lambda: fake_router(cloud))
    meeting_cli.main(["weekly", "--send"])
    out = capsys.readouterr().out
    assert "רונית שמעוני פתחה את הישיבה" in out  # names restored on this computer
    assert "איתי ביטון" in out and "2026-10-15" in out
    with repositories(academy) as repos:
        assert len(repos.proposals.list_by_meeting(1)) == 1


@pytest.mark.integration
def test_send_without_a_model_stops_before_recording_anything(academy, monkeypatch):
    broken = Router(models={**MODELS, "claude-haiku": ModelSpec(runs_locally=False, make_client=unavailable)})
    for cli in (meeting_cli, talk_cli):
        monkeypatch.setattr(cli, "get_router", lambda: broken)
    with pytest.raises(SystemExit, match="Cannot send"):
        meeting_cli.main(["weekly", "--send"])
    with pytest.raises(SystemExit, match="Cannot send"):
        talk_cli.main(["maya", "--send"])
    with repositories(academy) as repos:
        assert repos.meetings.list_all() == []
        assert repos.talks.list_by_trainee(22) == []


@pytest.mark.integration
def test_the_talk_dry_run_and_send(academy, monkeypatch, capsys):
    talk_cli.main(["maya"])
    dry = capsys.readouterr().out
    assert "Nothing was sent" in dry and no_known_name(academy, dry.split("```")[1])
    assert "רופא" not in dry and "removed before sending: 1" in dry  # exactly what the pipeline sends

    answer = {"summary": "[TRAINEE_22] רצה 3 ק״מ ברצף פעמיים השבוע, ומרוצה מההתקדמות.", "progress": [],
              "notes": [{"kind": "worked", "text": "יעד קטן ומדיד עוזר לה להתמיד"}]}
    cloud = FakeLLMClient(answers=[json.dumps(answer, ensure_ascii=False)], runs_locally=False)
    monkeypatch.setattr(talk_cli, "get_router", lambda: fake_router(cloud))
    talk_cli.main(["maya", "--send"])
    out = capsys.readouterr().out
    assert "מאיה גולן רצה 3" in out and "יעד קטן ומדיד" in out
    assert "רופא" not in cloud.calls[-1].user_message  # removed before sending
