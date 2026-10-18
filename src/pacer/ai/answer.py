"""Answering a staff member's question from the library, with a citation for every claim (RAG).

    retrieve      the passages closest in meaning (ai/library.py), on this computer
    anonymize     the QUESTION may name a trainee: names become tokens, as in every pipeline
    message       the question, and each passage with an id (P1, P2...) and what kind of source it is
    model         Claude answers only from the passages, as JSON: answer, claims with passage ids
    validate      every cited id must be a passage that was sent; a claim with no passage is reported
    citations     built from OUR metadata (work, section, pages, evidence level), never from the model

With no passage above the similarity floor, no model is asked at all: the answer is "the library has
no source for this". That is the cheapest and the most honest answer RAG can give.

The passages are library text, not personal data: classical sources name people ("רב יהודה", "שמואל")
who are not in the academy. So the final name check runs on the anonymized QUESTION, not on the passages.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from pacer.repositories import ScoredChunk, UnitOfWork

from .anonymization import AnonymizationError, AnonymizedText, Anonymizer, deanonymize, load_roster, tokens_in
from .embeddings import Embedder
from .issues import Issue, IssueCode
from .library import retrieve
from .prompts import Prompt, load_prompt
from .routing import AITask, Router
from .structured import complete_structured

log = logging.getLogger(__name__)

NO_SOURCE = "לא נמצא בספרייה מקור שעונה על השאלה הזו. אפשר לנסח אותה אחרת, או להוסיף לספרייה מקור מתאים."


class Coverage(StrEnum):
    """How well the passages answer the question. Three values, because "yes / no" hid the common case:
    a classical source about teachers of children does not answer a question about counselors directly,
    but applying it is a useful, honest answer, as long as it says that it is an application."""

    DIRECT = "direct"  # the passages answer the question itself
    PARTIAL = "partial"  # they answer a related or broader situation; the answer applies them, and says so
    NONE = "none"  # nothing in them helps: no claims, only what the library does and does not cover


class AnswerClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=3, max_length=600, description="One sentence of the answer")
    passages: list[str] = Field(max_length=6, description="The ids of the passages it rests on, e.g. P1")


class LibraryAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(max_length=5000, description="Hebrew, 2-6 short paragraphs")
    claims: list[AnswerClaim] = Field(max_length=20)
    coverage: Coverage = Field(description="direct, partial (applied from a related situation), or none")


@dataclass(frozen=True)
class Citation:
    passage: str  # "P1"
    source_id: int
    source: str  # "משנה תורה, הלכות תלמוד תורה פרק ב' (רמב"ם)"
    evidence: str  # "מקור ערכי-מסורתי"
    section: str | None
    pages: str | None
    similarity: float
    cited: bool  # did the answer actually rely on it


@dataclass(frozen=True)
class SupportedClaim:
    text: str
    passages: list[str]


@dataclass
class AskResult:
    question: str
    answer: str
    coverage: Coverage
    claims: list[SupportedClaim]
    citations: list[Citation]
    issues: list[Issue] = field(default_factory=list)
    prompt_ref: str | None = None  # None when no model was asked
    cost_usd: float | None = 0.0

    @property
    def insufficient(self) -> bool:
        """The library has no source for this question."""
        return self.coverage is Coverage.NONE


def passage_ids(hits: list[ScoredChunk]) -> list[str]:
    return [f"P{i}" for i in range(1, len(hits) + 1)]


def answer_message(question: AnonymizedText, hits: list[ScoredChunk]) -> str:
    """The question (anonymized: the type proves it) and the passages, each with its id and kind."""
    if not isinstance(question, AnonymizedText):
        raise TypeError("answer_message() needs AnonymizedText, from Anonymizer.anonymize()")
    lines = ["Question:", question.text, "", "Passages:"]
    for pid, hit in zip(passage_ids(hits), hits, strict=True):
        where = " · ".join(p for p in (hit.chunk.section, hit.chunk.pages) if p)
        lines += [
            "",
            f"[{pid}] {hit.source.citation} · {hit.source.evidence_level.label}" + (f" · {where}" if where else ""),
            hit.chunk.text,
        ]
    return "\n".join(lines)


_PASSAGE_ID = re.compile(r"^P\d{1,2}$")


def validate_answer(raw: LibraryAnswer, hits: list[ScoredChunk], question: AnonymizedText) -> AskResult:
    sent = set(passage_ids(hits))
    issues: list[Issue] = []
    known_tokens = set(question.people) | set(question.unresolved)

    def restore(text: str, item: int | None) -> str:
        if invented := sorted(set(tokens_in(text)) - known_tokens):
            issues.append(Issue(IssueCode.UNKNOWN_TOKEN, item, f"left as written: {', '.join(invented)}"))
        return deanonymize(text, question)

    claims: list[SupportedClaim] = []
    for number, claim in enumerate(raw.claims, start=1):
        ids = [p.strip() for p in claim.passages]
        unknown = [p for p in ids if not _PASSAGE_ID.match(p) or p not in sent]
        if unknown:
            issues.append(Issue(IssueCode.UNKNOWN_PASSAGE, number, f"cited {', '.join(unknown)}, never sent; removed"))
        valid = list(dict.fromkeys(p for p in ids if p in sent))
        if not valid:
            issues.append(Issue(IssueCode.UNSUPPORTED_CLAIM, number, "no passage behind this claim; not listed"))
            continue
        claims.append(SupportedClaim(restore(claim.text, number), valid))
    if raw.coverage is not Coverage.NONE and not claims:
        issues.append(Issue(IssueCode.NO_CITATIONS, None, "an answer without a single supported claim"))
    if raw.coverage is Coverage.NONE and claims:
        issues.append(Issue(IssueCode.CONTRADICTORY_COVERAGE, None, "says the library has nothing, yet cites it"))

    cited = {p for c in claims for p in c.passages}
    citations = [
        Citation(
            passage=pid,
            source_id=hit.source.key,
            source=hit.source.citation,
            evidence=hit.source.evidence_level.label,
            section=hit.chunk.section,
            pages=hit.chunk.pages,
            similarity=round(hit.similarity, 3),
            cited=pid in cited,
        )
        for pid, hit in zip(passage_ids(hits), hits, strict=True)
    ]
    return AskResult(
        question="",  # filled by the caller, with the original wording
        answer=restore(raw.answer, None),
        coverage=raw.coverage,
        claims=claims,
        citations=citations,
        issues=issues,
    )


class LibraryAnswerer:
    """uow: short transactions (roster, search). No transaction is open while a model works."""

    def __init__(
        self,
        uow: UnitOfWork,
        router: Router,
        embedder: Embedder,
        chunking: str,
        *,
        k: int = 5,
        prompt: Prompt | None = None,
    ) -> None:
        self._uow = uow
        self._router = router
        self._embedder = embedder
        self._chunking = chunking
        self._k = k
        self._prompt = prompt or load_prompt("library_answer", 3)  # pinned

    def prepare(self, question: str) -> tuple[AnonymizedText, list[ScoredChunk]]:
        """Everything before the model: anonymize the question, and retrieve passages for it.
        Retrieval uses the ANONYMIZED question too, so a name never reaches even the local model."""
        if not question.strip():
            raise ValueError("The question is empty")
        with self._uow() as repos:
            anonymizer = Anonymizer(load_roster(repos.staff, repos.trainees))
        safe = anonymizer.anonymize(question)
        if anonymizer.contains_known_name(safe.text):  # defense in depth, on the question
            raise AnonymizationError("A known name is in the question; nothing was sent")
        hits = retrieve(self._uow, self._embedder, self._chunking, safe.text, k=self._k, for_cloud_model=True)
        return safe, hits

    def ask(self, question: str) -> AskResult:
        safe, hits = self.prepare(question)
        if not hits:
            log.info("Library question: no passage above the floor; no model asked")
            return AskResult(question=question, answer=NO_SOURCE, coverage=Coverage.NONE, claims=[], citations=[])
        client = self._router.client_for(AITask.ANSWER_FROM_LIBRARY)
        max_tokens = self._router.route(AITask.ANSWER_FROM_LIBRARY).max_tokens
        result = complete_structured(
            client, self._prompt, answer_message(safe, hits), LibraryAnswer, max_tokens=max_tokens
        )
        answer = validate_answer(result.value, hits, safe)
        answer.question = question
        answer.prompt_ref = result.prompt_ref
        answer.cost_usd = result.cost_usd
        log.info(
            "Library question answered: passages=%s cited=%s claims=%s issues=%s coverage=%s",
            len(hits),
            sum(c.cited for c in answer.citations),
            len(answer.claims),
            len(answer.issues),
            answer.coverage,
        )
        return answer
