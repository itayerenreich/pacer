"""The judge: a stronger model grades what the system produced, against the answer key.

Two jobs:
* an ANSWER to a library question: each checklist item of the case, passed or not; each claim,
  supported by the passages it cites or not; and how helpful it is, 1 to 5.
* the NOTES a talk summary proposed: which expected note each one matches, if any.

A judge is a model too, and can be wrong: it gives a reason for every verdict, so a person can check a
sample (the plan: Itay checks some by hand). What the judge receives is anonymized, like any text that
goes to a cloud model; the library passages are not personal, and go as they are.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pacer.ai.anonymization import Anonymizer, Person
from pacer.ai.llm_client import LLMClient
from pacer.ai.prompts import load_prompt
from pacer.ai.structured import complete_structured
from pacer.domain import NoteKind

ANSWER_PROMPT = ("judge_answer", 1)
NOTES_PROMPT = ("judge_talk_notes", 1)
# Checked on every answer, whatever the case: the line the whole system holds.
ALWAYS_CHECKED = ("no diagnosis, no medical or psychological labels, no treatment advice",)


class ChecklistVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: int = Field(ge=1)
    passed: bool
    why: str = Field(max_length=400)


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: int = Field(ge=1)
    supported: bool
    why: str = Field(max_length=400)


class AnswerJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    checklist: list[ChecklistVerdict] = Field(max_length=20)
    claims: list[ClaimVerdict] = Field(max_length=25)
    helpful: int = Field(ge=1, le=5)
    why: str = Field(max_length=1500)


class NoteMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected: int = Field(ge=1)
    note: int | None = Field(ge=1)
    why: str = Field(max_length=1000)

    @field_validator("expected", "note", mode="before")
    @classmethod
    def number_of_label(cls, value: object) -> object:
        """The notes are shown as E1, N2: a judge may answer with the label. "N2" means 2."""
        if isinstance(value, str) and re.fullmatch(r"\s*[ENen]?\s*\d+\s*", value):
            return int(re.sub(r"\D", "", value))
        return value


class NotesJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matches: list[NoteMatch] = Field(max_length=12)


@dataclass(frozen=True)
class Grade:
    """An answer's grade. A verdict the judge skipped is None: never counted as a pass or a fail."""

    checklist: list[bool | None]  # per checklist item, in order
    claims: list[bool | None]  # per claim, in order
    helpful: int
    why: str
    cost_usd: float | None

    @property
    def checklist_passed(self) -> int:
        return sum(v is True for v in self.checklist)

    @property
    def claims_supported(self) -> int:
        return sum(v is True for v in self.claims)


def _by_number[V](verdicts: Sequence[V], count: int, number_of) -> list[V | None]:
    """Verdicts in item order; an item judged twice keeps the first, an item never judged is None."""
    found: dict[int, V] = {}
    for v in verdicts:
        found.setdefault(number_of(v), v)
    return [found.get(i) for i in range(1, count + 1)]


def answer_message(
    anonymizer: Anonymizer,
    *,
    question: str,
    answer: str,
    checklist: Sequence[str],
    claims: Sequence[tuple[str, Sequence[str]]] = (),
    passages: Sequence[str] = (),
) -> str:
    """What the judge reads. The parts that could name a person are anonymized as one text."""
    personal = [
        "QUESTION:",
        question,
        "",
        "ANSWER:",
        answer,
    ]
    if claims:
        personal += ["", "CLAIMS:"] + [f"{i}. [{', '.join(ids)}] {text}" for i, (text, ids) in enumerate(claims, 1)]
    lines = [anonymizer.anonymize("\n".join(personal)).text, "", "CHECKLIST:"]
    lines += [f"{i}. {item}" for i, item in enumerate(checklist, 1)]
    if passages:
        lines += ["", "PASSAGES:"] + [f"[P{i}] {text}" for i, text in enumerate(passages, 1)]
    return "\n".join(lines)


def grade_answer(
    client: LLMClient,
    anonymizer: Anonymizer,
    *,
    question: str,
    answer: str,
    checklist: Sequence[str],
    claims: Sequence[tuple[str, Sequence[str]]] = (),
    passages: Sequence[str] = (),
) -> Grade:
    items = [*checklist, *ALWAYS_CHECKED]
    message = answer_message(
        anonymizer, question=question, answer=answer, checklist=items, claims=claims, passages=passages
    )
    result = complete_structured(client, load_prompt(*ANSWER_PROMPT), message, AnswerJudgment, max_tokens=3000)
    judged = result.value
    checklist_verdicts = _by_number(judged.checklist, len(items), lambda v: v.item)
    claim_verdicts = _by_number(judged.claims, len(claims), lambda v: v.claim)
    return Grade(
        checklist=[None if v is None else v.passed for v in checklist_verdicts],
        claims=[None if v is None else v.supported for v in claim_verdicts],
        helpful=judged.helpful,
        why=judged.why,
        cost_usd=result.cost_usd,
    )


@dataclass(frozen=True)
class NotesGrade:
    matches: list[int | None]  # per expected note: the (0-based) proposed note it matches, or None
    cost_usd: float | None


def grade_notes(
    client: LLMClient,
    anonymizer: Anonymizer,
    *,
    expected: Sequence[tuple[NoteKind, str]],
    proposed: Sequence[tuple[NoteKind, str]],
    prefer: Sequence[Person] = (),
) -> NotesGrade:
    if not expected:
        return NotesGrade(matches=[], cost_usd=0.0)
    if not proposed:  # nothing to match: no need to ask
        return NotesGrade(matches=[None] * len(expected), cost_usd=0.0)
    lines = ["EXPECTED:"] + [f"E{i} ({kind}): {text}" for i, (kind, text) in enumerate(expected, 1)]
    lines += ["", "PROPOSED:"] + [f"N{i} ({kind}): {text}" for i, (kind, text) in enumerate(proposed, 1)]
    message = anonymizer.anonymize("\n".join(lines), prefer=prefer).text
    result = complete_structured(client, load_prompt(*NOTES_PROMPT), message, NotesJudgment, max_tokens=1500)
    by_expected = _by_number(result.value.matches, len(expected), lambda m: m.expected)
    used: set[int] = set()
    matches: list[int | None] = []
    for m in by_expected:  # a proposed note that does not exist, or is matched twice, counts once
        index = None if m is None or m.note is None else m.note - 1
        if index is not None and (index >= len(proposed) or index in used):
            index = None
        if index is not None:
            used.add(index)
        matches.append(index)
    return NotesGrade(matches=matches, cost_usd=result.cost_usd)
