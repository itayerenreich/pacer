"""Embeddings: a text as a vector of numbers, where texts with a similar MEANING get close vectors.

Embedding models are small and fast compared with language models, so in Pacer they run on this
computer, through Ollama: the library's text never leaves it to be embedded, and neither does a
counselor's question. Two open models are candidates; which one is used is decided by measuring
them on our own evaluation set (stage 4, task 8), not by reputation.

    embedder = ollama_embedder("bge-m3")
    vectors = embedder.embed_documents(["chunk one", "chunk two"])   # when indexing the library
    vector = embedder.embed_query("what do I do when...")            # for every question

Why two methods: some models are trained to see a QUESTION and a PASSAGE differently, and expect
the question to start with an instruction (Qwen3-Embedding does). The spec of each model says so;
code that searches never has to know.

The same model, with the same settings, must embed the library and the question. A vector of one
model means nothing to another: the name stored with every embedding is EmbeddingSpec.name.
"""

import logging
import math
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from pacer.config import get_ai_settings

from .llm_client import LLMError, litellm, translate_error

log = logging.getLogger(__name__)


class EmbeddingError(LLMError):
    """The model answered, but not with usable vectors (wrong count, size, or not numbers)."""


@dataclass(frozen=True)
class EmbeddingSpec:
    name: str  # stored with every vector; changing a setting below means a NEW name
    litellm_model: str
    dimensions: int
    query_prefix: str = ""
    document_prefix: str = ""
    max_words: int = 2_000  # a chunk longer than this is a bug in the chunking, not something to embed
    # Below this cosine similarity a chunk is not relevant to a question. Per model (each has its own
    # scale), calibrated on questions the library cannot answer (python -m pacer.library compare).
    min_similarity: float | None = None


# The candidates. Both are open, multilingual (Hebrew included), and run on a CPU.
EMBEDDING_MODELS: dict[str, EmbeddingSpec] = {
    spec.name: spec
    for spec in (
        # Measured 2026-10-18 (classics, 8 questions + 3 off-topic): hit@1 7/8 with fixed-200-40. The
        # lowest right source scored 0.469 (structure-220: 0.452), the highest off-topic hit 0.410 (0.394):
        # the floor sits in the gap. A narrow gap on a small set: measure again as the library grows.
        EmbeddingSpec(name="bge-m3", litellm_model="ollama/bge-m3", dimensions=1024, min_similarity=0.44),
        EmbeddingSpec(
            name="qwen3-embedding-0.6b",
            litellm_model="ollama/qwen3-embedding:0.6b",
            dimensions=1024,
            # No floor: off-topic questions scored as high as right sources (0.424 vs 0.400), nothing separates.
            # Qwen3-Embedding is trained with an instruction before the query, not before passages.
            query_prefix=(
                "Instruct: Given a question from a counselor at a pre-military academy, "
                "retrieve passages from education books and guidelines that answer it\nQuery: "
            ),
        ),
    )
}


class Embedder(Protocol):
    spec: EmbeddingSpec
    runs_locally: bool

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class LiteLLMEmbedder:
    """An embedding model through LiteLLM (Ollama, on this computer). Sends texts in batches."""

    def __init__(
        self, spec: EmbeddingSpec, *, runs_locally: bool, api_base: str | None = None, batch_size: int = 16
    ) -> None:
        self.spec = spec
        self.runs_locally = runs_locally
        self._api_base = api_base
        self._batch_size = batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = [self.spec.document_prefix + text for text in texts[start : start + self._batch_size]]
            vectors += self._embed(batch)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.spec.query_prefix + text])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        for text in texts:
            if not text.strip():
                raise ValueError("Cannot embed an empty text")
            if len(text.split()) > self.spec.max_words:
                raise ValueError(f"A text of {len(text.split())} words is too long to embed; split it first")
        started = time.perf_counter()
        try:
            raw = litellm.embedding(model=self.spec.litellm_model, input=texts, api_base=self._api_base, timeout=300)
        except Exception as error:  # every provider failure leaves as one of OUR errors
            translated = translate_error(error)
            log.warning("Embedding failed: model=%s error=%s", self.spec.name, type(translated).__name__)
            raise translated from error
        vectors = [item["embedding"] for item in raw.data]
        check_vectors(vectors, expected=len(texts), dimensions=self.spec.dimensions)
        log.info(
            "Embedded %s texts: model=%s latency=%sms",
            len(texts),
            self.spec.name,
            round((time.perf_counter() - started) * 1000),
        )
        return vectors


_NIQQUD = re.compile("[\u0591-\u05c7]")  # niqqud and cantillation marks (kept in the stored text)


def for_embedding(text: str, *, title: str | None = None, section: str | None = None) -> str:
    """The text as the embedding model sees it.

    * Without niqqud: "חֲנֹךְ לַנַּעַר" and "חנוך לנער" are the same words; marks would make them
      look different to the model. The stored chunk keeps them, for the citation.
    * With a short header naming the work and the section: a chunk of the Rambam that never says
      "Rambam" or "teacher" still carries that context into its vector ("contextual chunk header").
    """
    header = " · ".join(part for part in (title, section) if part)
    body = _NIQQUD.sub("", text)
    return f"{header}\n{body}" if header else body


def check_vectors(vectors: list[list[float]], *, expected: int, dimensions: int) -> None:
    """A model's answer is checked like any other: the right count, the right size, real numbers,
    and not all zeros (a zero vector has no direction, so cosine similarity is undefined)."""
    if len(vectors) != expected:
        raise EmbeddingError(f"Asked for {expected} vectors, got {len(vectors)}")
    for vector in vectors:
        if len(vector) != dimensions:
            raise EmbeddingError(f"Expected {dimensions} numbers per vector, got {len(vector)}")
        if not all(isinstance(x, int | float) and math.isfinite(x) for x in vector):
            raise EmbeddingError("A vector holds something that is not a finite number")
        if not any(vector):
            raise EmbeddingError("A vector of zeros has no direction")


DEFAULT_EMBEDDING_MODEL = "bge-m3"  # chosen by measurement, 2026-10-18 (stage 4 doc, part 11)


def ollama_embedder(name: str = DEFAULT_EMBEDDING_MODEL) -> LiteLLMEmbedder:
    """One of EMBEDDING_MODELS, served by Ollama on this computer."""
    if name not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown embedding model {name!r}; known: {', '.join(EMBEDDING_MODELS)}")
    return LiteLLMEmbedder(EMBEDDING_MODELS[name], runs_locally=True, api_base=get_ai_settings().ollama_base_url)
