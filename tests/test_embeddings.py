"""Embeddings, indexing and search: the adapter over LiteLLM (with litellm itself faked), the
checks on what a model returns, and indexing in rounds with no transaction open during the model."""

from types import SimpleNamespace

import pytest
from fake_llm import FakeEmbedder
from fakes import FakeUnitOfWork, empty_repos

import pacer.ai.embeddings as embeddings
from pacer.ai import LLMUnavailableError
from pacer.ai.embeddings import EMBEDDING_MODELS, EmbeddingError, LiteLLMEmbedder, check_vectors
from pacer.ai.library import index_library, search_library
from pacer.ai.routing import PrivacyViolation
from pacer.domain import Chunk, EvidenceLevel, KnowledgeSource, SourceKind, SourceLicense

CHUNKING = "fixed-300-50"

# --- the adapter, with litellm.embedding faked ----------------------------------------------------


@pytest.fixture
def calls(monkeypatch):
    seen: list[dict] = []

    def fake_embedding(**request):
        seen.append(request)
        dims = 4
        return SimpleNamespace(data=[{"embedding": [1.0, float(i), 0.5, 0.0][:dims]} for i, _ in enumerate(request["input"])])

    monkeypatch.setattr(embeddings.litellm, "embedding", fake_embedding)
    return seen


def small(name="small", **changes):
    return embeddings.EmbeddingSpec(name=name, litellm_model=f"ollama/{name}", dimensions=4, **changes)


def test_documents_are_sent_in_batches_with_the_document_prefix(calls):
    embedder = LiteLLMEmbedder(small(document_prefix="passage: "), runs_locally=True, batch_size=2)
    vectors = embedder.embed_documents(["א", "ב", "ג"])
    assert len(vectors) == 3
    assert [c["input"] for c in calls] == [["passage: א", "passage: ב"], ["passage: ג"]]
    assert calls[0]["model"] == "ollama/small"


def test_a_query_gets_the_query_prefix_and_not_the_document_one(calls):
    embedder = LiteLLMEmbedder(small(query_prefix="Query: ", document_prefix="passage: "), runs_locally=True)
    embedder.embed_query("מה עושים?")
    assert calls[0]["input"] == ["Query: מה עושים?"]


def test_qwen_is_told_it_is_searching_for_a_counselor_and_bge_is_not():
    assert EMBEDDING_MODELS["qwen3-embedding-0.6b"].query_prefix.startswith("Instruct:")
    assert EMBEDDING_MODELS["bge-m3"].query_prefix == ""
    assert all(spec.document_prefix == "" for spec in EMBEDDING_MODELS.values())


def test_empty_and_overlong_texts_are_refused_before_any_call(calls):
    embedder = LiteLLMEmbedder(small(max_words=3), runs_locally=True)
    with pytest.raises(ValueError, match="empty"):
        embedder.embed_documents(["  "])
    with pytest.raises(ValueError, match="too long"):
        embedder.embed_documents(["אחת שתיים שלוש ארבע"])
    assert calls == []


def test_a_provider_failure_becomes_our_error(monkeypatch):
    import litellm

    def down(**_):
        raise litellm.exceptions.APIConnectionError(message="refused", llm_provider="ollama", model="bge-m3")

    monkeypatch.setattr(embeddings.litellm, "embedding", down)
    with pytest.raises(LLMUnavailableError):
        LiteLLMEmbedder(small(), runs_locally=True).embed_query("שאלה")


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[1.0, 0.0, 0.0, 0.0]], "Asked for 2 vectors"),
        ([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], "Expected 4 numbers"),
        ([[1.0, 0.0, 0.0, float("nan")], [1.0, 0.0, 0.0, 0.0]], "finite"),
        ([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], "zeros"),
    ],
)
def test_what_the_model_returns_is_checked(vectors, message):
    with pytest.raises(EmbeddingError, match=message):
        check_vectors(vectors, expected=2, dimensions=4)


# --- indexing and search, on fakes -------------------------------------------------------------------

TEXTS = [
    "חניך שמתבודד מהקבוצה צריך שיחה אישית ותפקיד קטן",
    "הכנה לגיבוש בצבא דורשת כושר ריצה ושינה מסודרת",
    "סדר יום במכינה כולל לימוד התנדבות ושיחות קבוצה",
]


@pytest.fixture
def uow():
    repos = empty_repos()
    return FakeUnitOfWork(repos)


def add_source(uow, *, local_only=False, texts=TEXTS) -> int:
    with uow() as repos:
        source = repos.sources.add(
            KnowledgeSource(
                title="מקור לבדיקה",
                kind=SourceKind.INTERNAL,
                evidence_level=EvidenceLevel.PRACTICE,
                license=SourceLicense.INTERNAL,
                local_only=local_only,
            )
        )
        repos.chunks.add_many([Chunk(source_id=source.key, chunking=CHUNKING, ordinal=i, text=t) for i, t in enumerate(texts)])
    return source.key


def test_indexing_embeds_every_chunk_in_rounds_and_then_has_nothing_left(uow):
    add_source(uow)
    embedder = FakeEmbedder()
    report = index_library(uow, embedder, CHUNKING, batch_size=2)
    assert (report.embedded, report.batches) == (3, 2)
    assert index_library(uow, embedder, CHUNKING).embedded == 0  # running again does nothing


def test_no_transaction_is_open_while_the_model_embeds(uow):
    add_source(uow)

    class Watching(FakeEmbedder):
        def embed_documents(self, texts):
            assert uow.open == 0, "a transaction is open during the model call"
            return super().embed_documents(texts)

    index_library(uow, Watching(), CHUNKING)


def test_the_closest_chunk_in_meaning_comes_first(uow):
    add_source(uow)
    embedder = FakeEmbedder()
    index_library(uow, embedder, CHUNKING)
    hits = search_library(uow, embedder, CHUNKING, "מה עושים עם חניך שמתבודד מהקבוצה?", k=3)
    assert hits[0].chunk.text == TEXTS[0]
    assert hits[0].similarity > hits[1].similarity


def test_a_local_only_source_is_never_embedded_by_a_cloud_model_nor_sent_to_one(uow):
    add_source(uow, local_only=True)
    with pytest.raises(PrivacyViolation):
        index_library(uow, FakeEmbedder(runs_locally=False), CHUNKING)
    local = FakeEmbedder()
    index_library(uow, local, CHUNKING)
    assert search_library(uow, local, CHUNKING, "חניך שמתבודד", for_cloud_model=True) == []
    assert search_library(uow, local, CHUNKING, "חניך שמתבודד")  # for a local answer it is found


def test_a_chunk_deleted_while_the_model_worked_is_skipped(uow):
    source_id = add_source(uow)

    class Rechunking(FakeEmbedder):
        def embed_documents(self, texts):
            with uow() as repos:  # meanwhile, someone re-chunks the source
                repos.chunks.delete_by_source(source_id, CHUNKING)
            return super().embed_documents(texts)

    assert index_library(uow, Rechunking(), CHUNKING).embedded == 3  # embedded, but nothing saved
    with uow() as repos:
        assert repos.chunks.missing_embeddings("fake-embed", CHUNKING) == []


def test_two_models_keep_separate_vectors(uow):
    add_source(uow)
    a, b = FakeEmbedder("model-a"), FakeEmbedder("model-b", dimensions=32)
    index_library(uow, a, CHUNKING)
    assert search_library(uow, b, CHUNKING, "חניך") == []  # b has embedded nothing yet
    index_library(uow, b, CHUNKING)
    assert len(search_library(uow, b, CHUNKING, "חניך", k=5)) == 3


def test_an_empty_question_is_refused(uow):
    with pytest.raises(ValueError):
        search_library(uow, FakeEmbedder(), CHUNKING, "   ")


@pytest.mark.integration
def test_index_and_search_in_postgresql(engine):
    from pacer.repositories import repositories

    def db_uow():
        return repositories(engine)

    add_source(db_uow)
    embedder = FakeEmbedder()
    assert index_library(db_uow, embedder, CHUNKING, batch_size=2).embedded == 3
    hits = search_library(db_uow, embedder, CHUNKING, "חניך שמתבודד מהקבוצה, מה עושים?", k=2)
    assert hits[0].chunk.text == TEXTS[0] and len(hits) == 2
    assert 0 < hits[1].similarity < hits[0].similarity <= 1
