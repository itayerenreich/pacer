"""Tasks 5-6: retrieval rules, and answering from the library with a citation for every claim.

The geometry is set by hand: chunks get chosen vectors, and the question gets a chosen vector, so
every similarity in these tests is known exactly. The model is a fake that records what it was sent.
"""

import json
from collections.abc import Iterator

import pytest
from fake_llm import FakeLLMClient, fake_router
from fakes import FakeUnitOfWork, empty_repos
from fastapi.testclient import TestClient

from pacer.ai.anonymization import Anonymizer, Person, PersonKind, Roster
from pacer.ai.answer import NO_SOURCE, Coverage, LibraryAnswer, LibraryAnswerer, answer_message, validate_answer
from pacer.ai.embeddings import EmbeddingSpec
from pacer.ai.issues import IssueCode
from pacer.ai.library import retrieve
from pacer.domain import (
    Chunk,
    Cohort,
    EvidenceLevel,
    Group,
    KnowledgeSource,
    SourceKind,
    SourceLicense,
    StaffMember,
    StaffRole,
    Trainee,
)

CHUNKING = "fixed-200-40"


class VectorEmbedder:
    """Every question becomes the same chosen vector; records the questions it saw."""

    def __init__(self, query: list[float], *, floor: float | None = None) -> None:
        self.spec = EmbeddingSpec(name="vec", litellm_model="fake/vec", dimensions=len(query), min_similarity=floor)
        self.runs_locally = True
        self.query = query
        self.queries: list[str] = []

    def embed_documents(self, texts):
        raise AssertionError("not used: vectors are set by hand")

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self.query


def add(repos, title: str, chunks: list[tuple[str, list[float]]], *, evidence=EvidenceLevel.TRADITION, local_only=False):
    source = repos.sources.add(
        KnowledgeSource(
            title=title,
            author="רמב\"ם" if "רמב" in title or "משנה תורה" in title else None,
            kind=SourceKind.CLASSIC if evidence is EvidenceLevel.TRADITION else SourceKind.STUDY,
            evidence_level=evidence,
            license=SourceLicense.PUBLIC_DOMAIN,
            local_only=local_only,
        )
    )
    saved = repos.chunks.add_many(
        [Chunk(source_id=source.key, chunking=CHUNKING, ordinal=i, text=t, section=f"הלכה {i + 1}") for i, (t, _) in enumerate(chunks)]
    )
    repos.chunks.save_embeddings("vec", {c.key: v for c, (_, v) in zip(saved, chunks, strict=True)})
    return source


@pytest.fixture
def repos():
    repos = empty_repos()
    # a small academy, so the anonymizer has a roster
    repos.staff.add(StaffMember(full_name="רונית שמעוני", role=StaffRole.MANAGER))
    cohort = repos.cohorts.add(Cohort(name="מחזור", start_date="2026-09-01", end_date="2027-06-30"))
    group = repos.groups.add(Group(name="קבוצה", cohort_id=cohort.key, counselor_ids=[1]))
    repos.trainees.add(Trainee(first_name="אדל", last_name="אזולאי", cohort_id=cohort.key, group_id=group.key, primary_counselor_id=1))
    return repos


@pytest.fixture
def uow(repos):
    return FakeUnitOfWork(repos)


# --- retrieval (task 5) ------------------------------------------------------------------------------


def test_nothing_below_the_floor_counts(uow, repos):
    add(repos, "רמב\"ם", [("קרוב", [1.0, 0.0]), ("רחוק", [0.0, 1.0])])
    hits = retrieve(uow, VectorEmbedder([1.0, 0.1], floor=0.5), CHUNKING, "שאלה")
    assert [h.chunk.text for h in hits] == ["קרוב"]


def test_one_source_cannot_fill_every_place(uow, repos):
    """Hubness: a general source close to everything. At most 2 chunks per source by default."""
    add(repos, "חיבור כללי", [(f"כללי {i}", [1.0, 0.01 * i]) for i in range(4)])
    add(repos, "רמב\"ם", [("ממוקד", [0.9, 0.3])])
    hits = retrieve(uow, VectorEmbedder([1.0, 0.0]), CHUNKING, "שאלה", k=3)
    assert [h.source.title for h in hits] == ["חיבור כללי", "חיבור כללי", "רמב\"ם"]


def test_a_close_call_goes_to_stronger_evidence(uow, repos):
    add(repos, "מקור מסורתי", [("מסורת", [1.0, 0.0])])
    add(repos, "מחקר", [("מחקר", [1.0, 0.02])], evidence=EvidenceLevel.RESEARCH)  # 0.9998: a close call
    hits = retrieve(uow, VectorEmbedder([1.0, 0.0]), CHUNKING, "שאלה", k=2)
    assert [h.source.title for h in hits] == ["מחקר", "מקור מסורתי"]


def test_a_clear_winner_stays_first_whatever_its_evidence(uow, repos):
    add(repos, "מקור מסורתי", [("מסורת", [1.0, 0.0])])
    add(repos, "מחקר", [("מחקר", [0.7, 0.7])], evidence=EvidenceLevel.RESEARCH)  # 0.707: clearly further
    hits = retrieve(uow, VectorEmbedder([1.0, 0.0]), CHUNKING, "שאלה", k=2)
    assert [h.source.title for h in hits] == ["מקור מסורתי", "מחקר"]


# --- the message and validation (task 6) ---------------------------------------------------------------


def anonymized(text: str):
    return Anonymizer(Roster([Person(PersonKind.TRAINEE, 7, "אדל", "אזולאי")])).anonymize(text)


def test_the_message_has_the_anonymized_question_and_every_passage_with_its_kind(uow, repos):
    add(repos, "משנה תורה, הלכות תלמוד תורה פרק ב'", [("עשרים וחמשה תינוקות למלמד אחד", [1.0, 0.0])])
    hits = retrieve(uow, VectorEmbedder([1.0, 0.0]), CHUNKING, "שאלה")
    message = answer_message(anonymized("כמה חניכים כמו אדל אפשר ללוות?"), hits)
    assert "[TRAINEE_7]" in message and "אדל" not in message
    assert "[P1] משנה תורה, הלכות תלמוד תורה פרק ב' (רמב\"ם) · מקור ערכי-מסורתי · הלכה 1" in message
    with pytest.raises(TypeError):
        answer_message("raw text", hits)  # type: ignore[arg-type]


def hits_for(uow, repos):
    add(repos, "משנה תורה", [("עשרים וחמשה תינוקות למלמד אחד", [1.0, 0.0]), ("מושיבין מלמדי תינוקות", [0.9, 0.1])])
    return retrieve(uow, VectorEmbedder([1.0, 0.0]), CHUNKING, "שאלה")


def test_claims_cite_only_passages_that_were_sent(uow, repos):
    raw = LibraryAnswer(
        answer="ברמב\"ם נאמר שלמלמד אחד עשרים וחמישה תלמידים, ול[TRAINEE_7] זה רלוונטי.",
        claims=[
            {"text": "עד עשרים וחמישה למלמד אחד.", "passages": ["P1", "P9"]},
            {"text": "טענה בלי מקור.", "passages": ["P7"]},
        ],
        coverage="direct",
    )
    result = validate_answer(raw, hits_for(uow, repos), anonymized("ואדל?"))
    assert [(c.text, c.passages) for c in result.claims] == [("עד עשרים וחמישה למלמד אחד.", ["P1"])]
    assert [(i.code, i.item) for i in result.issues] == [
        (IssueCode.UNKNOWN_PASSAGE, 1),
        (IssueCode.UNKNOWN_PASSAGE, 2),
        (IssueCode.UNSUPPORTED_CLAIM, 2),
    ]
    assert "ולאדל אזולאי" in result.answer  # the token comes back as the name, here
    assert [(c.passage, c.cited, c.section) for c in result.citations] == [("P1", True, "הלכה 1"), ("P2", False, "הלכה 2")]


def test_an_answer_that_cites_nothing_is_reported_unless_it_says_it_cannot_answer(uow, repos):
    hits = hits_for(uow, repos)
    silent = validate_answer(LibraryAnswer(answer="תשובה", claims=[], coverage="direct"), hits, anonymized("ש"))
    honest = validate_answer(LibraryAnswer(answer="אין מקור", claims=[], coverage="none"), hits, anonymized("ש"))
    assert [i.code for i in silent.issues] == [IssueCode.NO_CITATIONS]
    assert honest.issues == [] and honest.insufficient


def test_a_partial_answer_is_an_answer_and_none_with_claims_is_reported(uow, repos):
    """The live test found it: Rambam on teachers of children, asked about counselors. Claude answered
    with citations and flagged "insufficient", because one flag meant both "nothing" and "only related"."""
    hits = hits_for(uow, repos)
    claims = [{"text": "ברמב\"ם: עשרים וחמישה למלמד.", "passages": ["P1"]}]
    partial = validate_answer(LibraryAnswer(answer="יישום", claims=claims, coverage="partial"), hits, anonymized("ש"))
    torn = validate_answer(LibraryAnswer(answer="אין, אבל", claims=claims, coverage="none"), hits, anonymized("ש"))
    assert partial.coverage is Coverage.PARTIAL and not partial.insufficient and partial.issues == []
    assert [i.code for i in torn.issues] == [IssueCode.CONTRADICTORY_COVERAGE]


# --- the whole flow --------------------------------------------------------------------------------------

GOOD = json.dumps(
    {"answer": "לפי הרמב\"ם, עשרים וחמישה למלמד.", "claims": [{"text": "עשרים וחמישה למלמד.", "passages": ["P1"]}], "coverage": "direct"},
    ensure_ascii=False,
)


def test_with_no_relevant_passage_no_model_is_asked(uow, repos):
    add(repos, "משנה תורה", [("רחוק", [0.0, 1.0])])
    cloud = FakeLLMClient(answers=[GOOD], runs_locally=False)
    result = LibraryAnswerer(uow, fake_router(cloud), VectorEmbedder([1.0, 0.0], floor=0.5), CHUNKING).ask("שאלה?")
    assert result.answer == NO_SOURCE and result.insufficient and result.prompt_ref is None
    assert cloud.calls == []


def test_a_name_in_the_question_reaches_neither_model(uow, repos):
    add(repos, "משנה תורה", [("עשרים וחמשה תינוקות למלמד אחד", [1.0, 0.0])])
    cloud = FakeLLMClient(answers=[GOOD], runs_locally=False, native_schema=True)
    embedder = VectorEmbedder([1.0, 0.0])
    result = LibraryAnswerer(uow, fake_router(cloud), embedder, CHUNKING).ask("אדל אזולאי מתקשה בקבוצה גדולה, מה עושים?")
    assert all("אדל" not in q for q in embedder.queries)  # not even the local embedding model
    assert "אדל" not in cloud.calls[0].user_message and "[TRAINEE_1]" in cloud.calls[0].user_message
    assert result.question == "אדל אזולאי מתקשה בקבוצה גדולה, מה עושים?"  # the person who asked sees their words
    assert result.prompt_ref == "library_answer@v3" and result.citations[0].cited


def test_a_local_only_source_never_goes_to_the_cloud_model(uow, repos):
    add(repos, "ספר פרטי", [("טקסט שלא יוצא מהמחשב", [1.0, 0.0])], local_only=True)
    cloud = FakeLLMClient(answers=[GOOD], runs_locally=False)
    result = LibraryAnswerer(uow, fake_router(cloud), VectorEmbedder([1.0, 0.0]), CHUNKING).ask("שאלה?")
    assert result.answer == NO_SOURCE and cloud.calls == []


def test_no_transaction_is_open_while_claude_answers(uow, repos):
    add(repos, "משנה תורה", [("עשרים וחמשה תינוקות למלמד אחד", [1.0, 0.0])])

    def answer(message: str) -> str:
        assert uow.open == 0
        return GOOD

    LibraryAnswerer(uow, fake_router(FakeLLMClient(answers=answer, runs_locally=False)), VectorEmbedder([1.0, 0.0]), CHUNKING).ask("שאלה?")


# --- the endpoint --------------------------------------------------------------------------------------


@pytest.fixture
def client(repos) -> Iterator[TestClient]:
    from pacer.api.app import create_app
    from pacer.api.dependencies import get_ai_router, get_db_engine, get_embedder, get_unit_of_work

    add(repos, "משנה תורה, הלכות תלמוד תורה פרק ב'", [("עשרים וחמשה תינוקות למלמד אחד", [1.0, 0.0])])
    app = create_app()
    app.dependency_overrides = {
        get_db_engine: lambda: (_ for _ in ()).throw(AssertionError("no database in these tests")),
        get_unit_of_work: lambda: FakeUnitOfWork(repos),
        get_ai_router: lambda: fake_router(FakeLLMClient(answers=[GOOD], runs_locally=False)),
        get_embedder: lambda: VectorEmbedder([1.0, 0.0]),
    }
    with TestClient(app) as test_client:
        yield test_client


def test_any_staff_member_can_ask_and_gets_citations_from_our_metadata(client):
    response = client.post("/library/ask", json={"question": "כמה חניכים למדריך אחד?"}, headers={"X-Staff-Id": "1"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["claims"] == [{"text": "עשרים וחמישה למלמד.", "passages": ["P1"]}]
    citation = body["citations"][0]
    assert citation["source"] == "משנה תורה, הלכות תלמוד תורה פרק ב' (רמב\"ם)"
    assert citation["evidence"] == "מקור ערכי-מסורתי" and citation["section"] == "הלכה 1"
    assert body["prompt"] == "library_answer@v3" and body["coverage"] == "direct" and not body["insufficient"]


def test_the_endpoint_needs_a_staff_member_and_a_real_question(client):
    assert client.post("/library/ask", json={"question": "כמה חניכים?"}).status_code == 401
    assert client.post("/library/ask", json={"question": "?"}, headers={"X-Staff-Id": "1"}).status_code == 422
