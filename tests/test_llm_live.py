"""Against REAL models. Skipped by default and in CI. On your computer:

    uv run pytest --run-llm -m llm            # needs ANTHROPIC_API_KEY in .env, and Ollama running

They check what fakes cannot: that the key works, that the local model is reachable,
and that the connection prompt really produces a short Hebrew answer.
"""

import re

import pytest

from pacer.ai import load_prompt
from pacer.ai.clients import claude_client, ollama_client

pytestmark = pytest.mark.llm
HEBREW = re.compile(r"[֐-׿]")


@pytest.mark.parametrize("make_client", [claude_client, ollama_client], ids=["claude", "ollama"])
def test_model_answers_in_hebrew(make_client):
    response = make_client().complete(load_prompt("connection_check"), "שלום, מה שלומך?", max_tokens=150)

    assert HEBREW.search(response.text), response.text
    assert response.output_tokens > 0
    assert response.prompt_ref == "connection_check@v1"


def test_claude_reports_a_real_cost():
    response = claude_client().complete(load_prompt("connection_check"), "שלום", max_tokens=50)
    assert 0 < response.cost_usd < 0.01  # a greeting costs a tiny fraction of a cent


def test_claude_returns_a_valid_structured_object_through_the_router():
    from pacer.ai.checks import SAMPLE_TEXT, TextDigest
    from pacer.ai.routing import AITask, Router
    from pacer.ai.structured import complete_structured

    router = Router()
    result = complete_structured(
        router.client_for(AITask.STRUCTURED_CHECK),
        load_prompt("structured_check"),
        SAMPLE_TEXT,
        TextDigest,
        max_tokens=router.route(AITask.STRUCTURED_CHECK).max_tokens,
    )
    assert result.value.language == "he"
    assert HEBREW.search(result.value.summary)
    assert 1 <= len(result.value.topics) <= 3


# --- the meeting pipeline, with Claude (task 6) ------------------------------------------------------


class _Spy:
    """Wraps the real Claude client and keeps what was sent and what came back, for the asserts."""

    def __init__(self, client) -> None:
        self._client = client
        self.model, self.runs_locally, self.native_schema = client.model, client.runs_locally, client.native_schema
        self.sent: list[str] = []
        self.received: list[str] = []

    def complete(self, prompt, user_message, **options):
        self.sent.append(user_message)
        response = self._client.complete(prompt, user_message, **options)
        self.received.append(response.text)
        return response


def _run_sample(sample):
    from fakes import FakeUnitOfWork, empty_repos

    from pacer.ai.anonymization import Anonymizer, load_roster
    from pacer.ai.meeting_pipeline import MeetingPipeline
    from pacer.ai.routing import MODELS, ModelSpec, Router
    from pacer.demo import build_academy
    from pacer.services import MeetingService

    repos = empty_repos()
    build_academy(repos)
    spy = _Spy(claude_client("claude-haiku-5-5"))
    router = Router(models={**MODELS, "claude-haiku": ModelSpec(runs_locally=False, make_client=lambda: spy)})
    manager = repos.staff.get(1)
    meeting = MeetingService(repos.meetings, repos.proposals, repos.tasks, repos.staff).create_meeting(
        manager,
        title=sample.title,
        held_on=sample.held_on,
        participant_ids=sample.participant_ids,
        transcript=sample.transcript,
    )
    result = MeetingPipeline(FakeUnitOfWork(repos), router).process(manager, meeting.id)
    anonymizer = Anonymizer(load_roster(repos.staff, repos.trainees))
    assert not any(anonymizer.contains_known_name(message) for message in spy.sent)  # nothing left here
    return result, spy


def test_claude_extracts_the_weekly_meeting():
    from datetime import date

    from pacer.demo.meetings import WEEKLY

    result, _ = _run_sample(WEEKLY)
    by_owner = {p.owner_id: p for p in result.proposals}

    assert 3 <= len(result.proposals) <= 6, [p.description for p in result.proposals]
    assert {p.owner_id for p in result.proposals} <= {1, 2, 3, 4, 5, None}
    assert by_owner[5].due_date == date(2026, 10, 15)  # איתי, transport, "עד יום חמישי"
    assert by_owner[4].due_date == date(2026, 10, 18)  # שירה, seminar, "עד יום ראשון"
    assert "אדל אזולאי" in by_owner[3].description  # עומר's talk with אדל, name restored here
    assert not any("האקתון" in p.description for p in result.proposals)  # an idea, not agreed
    assert all(HEBREW.search(p.description) for p in result.proposals)
    assert HEBREW.search(result.meeting.summary)
    assert result.cost_usd is not None and result.cost_usd < 0.01


def test_claude_gives_a_task_to_an_absent_counselor_when_told_to():
    """In SHORT, the manager says "ושירה תכין את חוברת ההדרכה לטיול, אני אעדכן אותה" and שירה
    ([STAFF_4]) is absent. Since meeting_extract@v2 that task is hers, flagged for the manager.
    (v1 forbade it, and this live test showed the rule was wrong: decision 2026-10-15.)"""
    from pacer.ai.issues import IssueCode
    from pacer.demo.meetings import SHORT

    result, _ = _run_sample(SHORT)
    assert any(p.owner_id == 4 for p in result.proposals), [(p.owner_id, p.description) for p in result.proposals]
    assert any(i.code is IssueCode.OWNER_ABSENT for i in result.issues)
    assert any(p.owner_id == 3 for p in result.proposals)  # עומר: check which זיו
    assert all("[STAFF_" not in p.description for p in result.proposals)


# --- the personal talk pipeline, with Claude (task 7) ------------------------------------------------


def test_claude_summarizes_the_sample_talk_safely():
    from fakes import FakeUnitOfWork, empty_repos

    from pacer.ai.anonymization import Anonymizer, load_roster
    from pacer.ai.routing import MODELS, ModelSpec, Router
    from pacer.ai.talk_pipeline import TalkPipeline, is_sensitive, talk_service
    from pacer.demo import build_academy
    from pacer.demo.talks import MAYA
    from pacer.domain import TalkProposalKind

    repos = empty_repos()
    build_academy(repos)
    spy = _Spy(claude_client("claude-haiku-5-5"))
    router = Router(models={**MODELS, "claude-haiku": ModelSpec(runs_locally=False, make_client=lambda: spy)})
    itai = repos.staff.get(MAYA.counselor_id)
    uow = FakeUnitOfWork(repos)
    with uow() as r:
        talk = talk_service(r).record_talk(itai, MAYA.trainee_id, held_on=MAYA.held_on, notes=MAYA.notes)
    result = TalkPipeline(uow, router).summarize(itai, talk.id)

    assert not Anonymizer(load_roster(repos.staff, repos.trainees)).contains_known_name(spy.sent[0])
    assert "רופא" not in spy.sent[0] and result.removed_sentences == 1  # removed before sending
    progress = [p for p in result.proposals if p.kind is TalkProposalKind.PROGRESS]
    notes = [p for p in result.proposals if p.kind is TalkProposalKind.NOTE]
    assert 63 in {p.goal_id for p in progress}  # running: 2 -> 3 km, out of 5
    assert all(p.progress > 25 for p in progress if p.goal_id == 63)
    assert 1 <= len(notes) <= 5
    assert not any(is_sensitive(n.text) for n in notes)  # headaches and the doctor stay out of memory
    raw = spy.received[-1]
    assert "כאבי ראש" not in raw[raw.find('"notes"') :]  # and Claude did not even propose them
    assert HEBREW.search(result.talk.summary)
    assert result.cost_usd is not None and result.cost_usd < 0.01


# --- embeddings, with the real models in Ollama (stage 4) -------------------------------------------


@pytest.mark.parametrize("name", ["bge-m3", "qwen3-embedding-0.6b"])
def test_a_real_embedding_model_works_on_hebrew(name):
    """Needs `ollama pull bge-m3` / `ollama pull qwen3-embedding:0.6b`. Checks the MECHANICS: real
    vectors of the declared size, the same text twice gives the same vector, and a reworded
    sentence is closer than an unrelated one.

    Which model understands Hebrew counselors' questions BETTER is not a test with one example: on
    2026-10-18 bge-m3 put "חניך שמתכונן לגיבוש" (shares the word חניך) above "מתבודד מהחבר'ה"
    (shares only the meaning), and qwen3 did not. One example decides nothing; it is a question
    for the evaluation set (python -m pacer.library compare), where many questions decide."""
    from pacer.ai.embeddings import EMBEDDING_MODELS, ollama_embedder

    embedder = ollama_embedder(name)
    same, reworded, unrelated = embedder.embed_documents(
        ["חניך שיושב לבד בארוחות ולא מדבר עם הקבוצה",
         "חניך שמתבודד ולא משתלב עם החבר'ה בקבוצה",
         "לוח הזמנים של ההסעה לטיול השנתי"]
    )

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / ((sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5))

    assert len(same) == EMBEDDING_MODELS[name].dimensions
    assert cosine(same, embedder.embed_documents(["חניך שיושב לבד בארוחות ולא מדבר עם הקבוצה"])[0]) > 0.999
    assert cosine(same, reworded) > cosine(same, unrelated)


# --- answering from the library, with bge-m3 and Claude (stage 4, task 6) ----------------------------


def test_claude_answers_from_the_library_and_says_so_when_it_cannot():
    """Needs ANTHROPIC_API_KEY and `ollama pull bge-m3`. Two Rambam passages, embedded for real."""
    from fakes import FakeUnitOfWork, empty_repos

    from pacer.ai.answer import LibraryAnswerer
    from pacer.ai.embeddings import ollama_embedder
    from pacer.ai.library import index_library
    from pacer.ai.routing import get_router
    from pacer.domain import Chunk, EvidenceLevel, KnowledgeSource, SourceKind, SourceLicense

    repos = empty_repos()
    uow = FakeUnitOfWork(repos)
    with uow() as r:
        source = r.sources.add(
            KnowledgeSource(title="משנה תורה, הלכות תלמוד תורה פרק ב'", author='רמב"ם', kind=SourceKind.CLASSIC,
                            evidence_level=EvidenceLevel.TRADITION, license=SourceLicense.PUBLIC_DOMAIN)
        )
        r.chunks.add_many([
            Chunk(source_id=source.key, chunking="fixed-200-40", ordinal=0, section="הלכה ה",
                  text="עשרים וחמשה תינוקות למלמד אחד. היו יתר על עשרים וחמשה עד ארבעים, מושיבין עמו אחר לסייעו בלימודן."),
            Chunk(source_id=source.key, chunking="fixed-200-40", ordinal=1, section="הלכה ב",
                  text="מכניסין את התינוקות להתלמד כבן שש כבן שבע, לפי כח הבן ובנין גופו."),
        ])
    embedder = ollama_embedder("bge-m3")
    index_library(uow, embedder, "fixed-200-40")
    answerer = LibraryAnswerer(uow, get_router(), embedder, "fixed-200-40")

    result = answerer.ask("כמה חניכים יכול מדריך אחד ללוות, ומתי צריך עוד איש צוות?")
    # The passages speak of teachers of children, so "partial" (an application) is as right as "direct"
    assert not result.insufficient and result.issues == [], (result.coverage, result.issues)
    assert any(c.passage == "P1" and c.cited and c.section == "הלכה ה" for c in result.citations) or any(
        c.cited and c.section == "הלכה ה" for c in result.citations
    )
    assert HEBREW.search(result.answer)

    off_topic = answerer.ask("איך ממלאים טופס בקשה לדחיית שירות בצה\"ל?")
    assert off_topic.insufficient  # no source: either nothing passed the floor, or Claude said so
