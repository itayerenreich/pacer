"""Persistence for the knowledge library: sources, their chunks, and the chunks' embeddings.

Search is done by PostgreSQL itself, with pgvector: `embedding <=> query` is the cosine DISTANCE
(0 = same direction, 2 = opposite), so similarity = 1 - distance. The column has no fixed
dimension (we compare models with different sizes), which means no vector index: PostgreSQL
compares the query with every embedding of that model. For a few thousand chunks that is a few
milliseconds; an HNSW index can be added per model if the library grows a lot.
"""

from typing import cast

from sqlalchemy import ColumnElement, delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from pacer.domain import Chunk, KnowledgeSource

from ._mapping import row_values
from .interfaces import ScoredChunk
from .tables import ChunkEmbeddingRow, ChunkRow, KnowledgeSourceRow


def _source(row: KnowledgeSourceRow) -> KnowledgeSource:
    return KnowledgeSource.model_validate(row, from_attributes=True)


def _chunk(row: ChunkRow) -> Chunk:
    return Chunk.model_validate(row, from_attributes=True)


class KnowledgeSourceRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add(self, source: KnowledgeSource) -> KnowledgeSource:
        row = KnowledgeSourceRow(**row_values(source))
        self._session.add(row)
        self._session.flush()
        return source.model_copy(update={"id": row.id})

    def get(self, source_id: int) -> KnowledgeSource | None:
        row = self._session.get(KnowledgeSourceRow, source_id)
        return None if row is None else _source(row)

    def list_all(self) -> list[KnowledgeSource]:
        return [_source(r) for r in self._session.scalars(select(KnowledgeSourceRow).order_by(KnowledgeSourceRow.id))]

    def delete(self, source_id: int) -> None:
        """The source, its chunks and their embeddings (ON DELETE CASCADE)."""
        self._session.execute(delete(KnowledgeSourceRow).where(KnowledgeSourceRow.id == source_id))
        self._session.flush()


class ChunkRepository:
    def __init__(self, session: Session) -> None:
        self._session = session

    def add_many(self, chunks: list[Chunk]) -> list[Chunk]:
        rows = [ChunkRow(**row_values(chunk)) for chunk in chunks]
        self._session.add_all(rows)
        self._session.flush()
        return [chunk.model_copy(update={"id": row.id}) for chunk, row in zip(chunks, rows, strict=True)]

    def get_many(self, chunk_ids: list[int]) -> list[Chunk]:
        rows = self._session.scalars(select(ChunkRow).where(ChunkRow.id.in_(chunk_ids)))
        by_id = {row.id: _chunk(row) for row in rows}
        return [by_id[i] for i in chunk_ids if i in by_id]  # in the order asked for

    def list_by_source(self, source_id: int, chunking: str) -> list[Chunk]:
        rows = self._session.scalars(
            select(ChunkRow)
            .where(ChunkRow.source_id == source_id, ChunkRow.chunking == chunking)
            .order_by(ChunkRow.ordinal)
        )
        return [_chunk(row) for row in rows]

    def delete_by_source(self, source_id: int, chunking: str) -> int:
        """Re-chunking a source replaces its chunks of that strategy (embeddings go with them)."""
        result = self._session.execute(
            delete(ChunkRow).where(ChunkRow.source_id == source_id, ChunkRow.chunking == chunking)
        )
        self._session.flush()
        return int(result.rowcount)  # type: ignore[attr-defined]

    def save_embeddings(self, model: str, vectors: dict[int, list[float]]) -> None:
        """Insert or replace: embedding the same chunk again with the same model overwrites it."""
        if not vectors:
            return
        statement = insert(ChunkEmbeddingRow).values(
            [{"chunk_id": chunk_id, "model": model, "embedding": vector} for chunk_id, vector in vectors.items()]
        )
        self._session.execute(
            statement.on_conflict_do_update(
                index_elements=["chunk_id", "model"], set_={"embedding": statement.excluded.embedding}
            )
        )
        self._session.flush()

    def missing_embeddings(self, model: str, chunking: str) -> list[Chunk]:
        """Chunks of this strategy that this model has not embedded yet: indexing can resume."""
        embedded = select(ChunkEmbeddingRow.chunk_id).where(ChunkEmbeddingRow.model == model)
        rows = self._session.scalars(
            select(ChunkRow)
            .where(ChunkRow.chunking == chunking, ChunkRow.id.not_in(embedded))
            .order_by(ChunkRow.source_id, ChunkRow.ordinal)
        )
        return [_chunk(row) for row in rows]

    def search(
        self, model: str, chunking: str, query: list[float], *, k: int, cloud_safe_only: bool = False
    ) -> list[ScoredChunk]:
        """The k chunks closest to the query vector, best first.
        cloud_safe_only: skip sources marked local_only (their text may not go to a cloud model)."""
        # pgvector's comparator is untyped; say what it returns, so the whole query stays typed
        distance = cast(ColumnElement[float], ChunkEmbeddingRow.embedding.cosine_distance(query))
        conditions = [ChunkEmbeddingRow.model == model, ChunkRow.chunking == chunking]
        if cloud_safe_only:
            conditions.append(KnowledgeSourceRow.local_only.is_(False))
        statement = (
            select(ChunkRow, KnowledgeSourceRow, distance.label("distance"))
            .join(ChunkEmbeddingRow, ChunkEmbeddingRow.chunk_id == ChunkRow.id)
            .join(KnowledgeSourceRow, KnowledgeSourceRow.id == ChunkRow.source_id)
            .where(*conditions)
            .order_by(distance, ChunkRow.id)
            .limit(k)
        )
        return [
            ScoredChunk(chunk=_chunk(chunk), source=_source(source), similarity=1.0 - float(dist))
            for chunk, source, dist in self._session.execute(statement)
        ]
