"""The knowledge library: domain rules, and storage and search.

The storage tests run twice, against the in-memory fake and against PostgreSQL with pgvector
(integration), so the fake the fast tests rely on cannot drift from the real thing.
"""

from datetime import UTC, datetime

import pytest
from fakes import empty_repos

from pacer.domain import Chunk, EvidenceLevel, KnowledgeSource, SourceKind, SourceLicense
from pacer.repositories import repositories

MODEL = "test-embed"


def source(**changes) -> KnowledgeSource:
    fields = {
        "title": "הדרכה בקבוצה",
        "author": "ישראל ישראלי",
        "year": 2021,
        "kind": SourceKind.BOOK,
        "evidence_level": EvidenceLevel.PROFESSIONAL,
        "license": SourceLicense.OWNED_COPY,
    }
    return KnowledgeSource(**{**fields, **changes})


def chunk(source_id: int, ordinal: int, text: str = "חניך שמתבודד מהקבוצה", **changes) -> Chunk:
    return Chunk(source_id=source_id, chunking="fixed-300-50", ordinal=ordinal, text=text, **changes)


# --- domain -----------------------------------------------------------------------------------------


def test_evidence_levels_are_ranked_strongest_first():
    ranks = [level.rank for level in EvidenceLevel]
    assert ranks == sorted(ranks, reverse=True) and EvidenceLevel.SYSTEMATIC.rank == 4


def test_a_citation_names_the_source_with_what_is_known():
    assert source().citation == "הדרכה בקבוצה (ישראל ישראלי, 2021)"
    assert source(author=None, year=None).citation == "הדרכה בקבוצה"


def test_pages_must_come_in_pairs_and_in_order():
    assert chunk(1, 0, page_start=45, page_end=46).pages == "עמ׳ 45–46"
    assert chunk(1, 0, page_start=45, page_end=45).pages == "עמ׳ 45"
    assert chunk(1, 0).pages is None
    with pytest.raises(ValueError):
        chunk(1, 0, page_start=46, page_end=45)
    with pytest.raises(ValueError):
        chunk(1, 0, page_start=46)


def test_a_source_needs_a_title_and_a_sensible_year():
    with pytest.raises(ValueError):
        source(title="")
    with pytest.raises(ValueError):
        source(year=1700)


# --- storage and search: the same tests on the fake and on PostgreSQL --------------------------------


@pytest.fixture(params=["fake", pytest.param("postgres", marks=pytest.mark.integration)])
def uow(request):
    """A factory of units of work, over fakes or over the test database."""
    if request.param == "fake":
        repos = empty_repos()
        from fakes import FakeUnitOfWork

        return FakeUnitOfWork(repos)
    engine = request.getfixturevalue("engine")
    return lambda: repositories(engine)


def library(uow, *, local_only: bool = False) -> tuple[int, list[int]]:
    """A source with three chunks, embedded in 3 dimensions so the geometry is easy to see."""
    vectors = [[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]]
    with uow() as repos:
        src = repos.sources.add(source(local_only=local_only))
        saved = repos.chunks.add_many([chunk(src.key, i, f"קטע מספר {i}") for i in range(3)])
        repos.chunks.save_embeddings(MODEL, {c.key: v for c, v in zip(saved, vectors, strict=True)})
    return src.key, [c.key for c in saved]


def test_a_source_and_its_chunks_round_trip(uow):
    source_id, ids = library(uow)
    with uow() as repos:
        assert repos.sources.get(source_id).citation == "הדרכה בקבוצה (ישראל ישראלי, 2021)"
        assert [c.text for c in repos.chunks.list_by_source(source_id, "fixed-300-50")] == [
            "קטע מספר 0",
            "קטע מספר 1",
            "קטע מספר 2",
        ]
        assert [c.key for c in repos.chunks.get_many([ids[2], ids[0]])] == [ids[2], ids[0]]  # order kept
        assert repos.chunks.list_by_source(source_id, "other") == []


def test_search_returns_the_closest_chunks_first_with_cosine_similarity(uow):
    _, ids = library(uow)
    with uow() as repos:
        hits = repos.chunks.search(MODEL, "fixed-300-50", [1.0, 0.1, 0.0], k=2)
    assert [h.chunk.key for h in hits] == [ids[0], ids[1]]
    assert hits[0].similarity == pytest.approx(0.995, abs=0.001)
    assert hits[1].similarity == pytest.approx(0.855, abs=0.001)
    assert hits[0].source.title == "הדרכה בקבוצה"


def test_search_only_looks_at_one_model_and_one_chunking(uow):
    _, ids = library(uow)
    with uow() as repos:
        assert repos.chunks.search("another-model", "fixed-300-50", [1.0, 0.0, 0.0], k=5) == []
        assert repos.chunks.search(MODEL, "structure-400", [1.0, 0.0, 0.0], k=5) == []


def test_a_local_only_source_can_be_left_out_of_a_search_for_a_cloud_model(uow):
    library(uow, local_only=True)
    with uow() as repos:
        assert len(repos.chunks.search(MODEL, "fixed-300-50", [1.0, 0.0, 0.0], k=5)) == 3
        assert repos.chunks.search(MODEL, "fixed-300-50", [1.0, 0.0, 0.0], k=5, cloud_safe_only=True) == []


def test_embedding_again_replaces_and_missing_embeddings_lets_indexing_resume(uow):
    source_id, ids = library(uow)
    with uow() as repos:
        extra = repos.chunks.add_many([chunk(source_id, 3, "קטע שעוד לא עבר embedding")])
        assert [c.key for c in repos.chunks.missing_embeddings(MODEL, "fixed-300-50")] == [extra[0].key]
        repos.chunks.save_embeddings(MODEL, {ids[2]: [1.0, 0.0, 0.0]})  # chunk 2 moves next to the query
    with uow() as repos:
        top = repos.chunks.search(MODEL, "fixed-300-50", [1.0, 0.0, 0.0], k=1)
        assert top[0].similarity == pytest.approx(1.0)


def test_deleting_a_source_deletes_its_chunks_and_embeddings(uow):
    source_id, _ = library(uow)
    with uow() as repos:
        repos.sources.delete(source_id)
    with uow() as repos:
        assert repos.sources.get(source_id) is None
        assert repos.chunks.list_by_source(source_id, "fixed-300-50") == []
        assert repos.chunks.search(MODEL, "fixed-300-50", [1.0, 0.0, 0.0], k=5) == []


def test_rechunking_replaces_the_chunks_of_that_strategy_only(uow):
    source_id, _ = library(uow)
    with uow() as repos:
        repos.chunks.add_many([Chunk(source_id=source_id, chunking="structure-400", ordinal=0, text="פרק 1")])
        assert repos.chunks.delete_by_source(source_id, "fixed-300-50") == 3
        assert [c.text for c in repos.chunks.list_by_source(source_id, "structure-400")] == ["פרק 1"]


def test_two_chunks_cannot_take_the_same_position(uow):
    source_id, _ = library(uow)
    with pytest.raises(Exception, match="uq_chunks_position"), uow() as repos:
        repos.chunks.add_many([chunk(source_id, 0, "כפילות")])


def test_hebrew_text_and_the_time_survive_the_round_trip(uow):
    text = "״מה שעבד: יעד קטן ומדיד״ – עמ׳ 12"
    with uow() as repos:
        src = repos.sources.add(source(added_at=datetime(2026, 10, 18, 15, 0, tzinfo=UTC)))
        repos.chunks.add_many([chunk(src.key, 0, text)])
    with uow() as repos:
        assert repos.chunks.list_by_source(src.key, "fixed-300-50")[0].text == text
        assert repos.sources.get(src.key).added_at == datetime(2026, 10, 18, 15, 0, tzinfo=UTC)


def test_a_classical_source_is_not_ranked_as_evidence():
    """Decision 2026-10-18: the Rambam is a source of values, outside the empirical ladder."""
    assert EvidenceLevel.TRADITION.rank == 0 and not EvidenceLevel.TRADITION.is_empirical
    assert EvidenceLevel.TRADITION.label == "מקור ערכי-מסורתי"
    assert all(level.is_empirical for level in EvidenceLevel if level is not EvidenceLevel.TRADITION)
