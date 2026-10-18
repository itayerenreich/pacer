"""Indexing the library and searching it: chunks become vectors, questions find chunks.

The same rule as the stage 3 pipelines: no model call inside a database transaction. Indexing
works in rounds: read a batch of chunks that this model has not embedded yet (short transaction),
embed them (no transaction: on a CPU this takes seconds), save the vectors (short transaction).
A crash or Ctrl+C loses at most one batch, and running again continues where it stopped, because
it only ever asks for the chunks that are still missing.

The privacy rule from the router applies here too: a source marked local_only is embedded only by
a model that runs on this computer, and a search whose results will go to a cloud model leaves it out.
"""

import logging
import time
from dataclasses import dataclass

from pacer.repositories import ScoredChunk, UnitOfWork

from .embeddings import Embedder, for_embedding
from .routing import PrivacyViolation

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexReport:
    model: str
    chunking: str
    embedded: int
    batches: int
    seconds: float


def index_library(uow: UnitOfWork, embedder: Embedder, chunking: str, *, batch_size: int = 32) -> IndexReport:
    """Embed every chunk of this chunking strategy that this model has not embedded yet."""
    model = embedder.spec.name
    started = time.perf_counter()
    embedded = batches = 0
    while True:
        # 1. Read, in a short transaction: the next batch still missing.
        with uow() as repos:
            batch = repos.chunks.missing_embeddings(model, chunking)[:batch_size]
            sources = {s.key: s for s in repos.sources.list_all()}
        if not batch:
            break
        if not embedder.runs_locally and any(sources[c.source_id].local_only for c in batch):
            raise PrivacyViolation("A local_only source may only be embedded by a model on this computer")
        # 2. The model, with NO transaction open. Each chunk with the name of its work, no niqqud.
        texts = [for_embedding(c.text, title=sources[c.source_id].title, section=c.section) for c in batch]
        vectors = embedder.embed_documents(texts)
        # 3. Write, in a short transaction. A chunk deleted meanwhile (re-chunked) is skipped.
        with uow() as repos:
            still_there = {chunk.key for chunk in repos.chunks.get_many([chunk.key for chunk in batch])}
            repos.chunks.save_embeddings(
                model,
                {chunk.key: vector for chunk, vector in zip(batch, vectors, strict=True) if chunk.key in still_there},
            )
        embedded += len(batch)
        batches += 1
        log.info("Indexed %s chunks (%s so far) with %s", len(batch), embedded, model)
    report = IndexReport(model, chunking, embedded, batches, round(time.perf_counter() - started, 2))
    log.info("Indexing done: %s", report)
    return report


def search_library(
    uow: UnitOfWork,
    embedder: Embedder,
    chunking: str,
    question: str,
    *,
    k: int = 5,
    for_cloud_model: bool = False,
) -> list[ScoredChunk]:
    """The k chunks closest in meaning to the question, best first.
    for_cloud_model: the hits will be sent to a model in the cloud, so local_only sources are left out."""
    if not question.strip():
        raise ValueError("The question is empty")
    query = embedder.embed_query(for_embedding(question))  # no transaction open
    with uow() as repos:
        return repos.chunks.search(embedder.spec.name, chunking, query, k=k, cloud_safe_only=for_cloud_model)
