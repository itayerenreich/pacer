"""Running the evaluation sets against real models, and scoring what came back.

LIBRARY, per case:
    retrieval    is a right source among the passages?             (code, no model)
    coverage     is the answer's coverage one of the acceptable?   (code)
    citations    does the answer cite a right source?              (code)
    claims       does each claim's passage really say it?          (judge)
    checklist    the case's "must / must not" items                (judge)
    helpful      1 to 5                                            (judge)
    baseline     the same question, same model, NO passages: graded on the same checklist. The
                 difference is what the library adds (or does not).

TALKS, per case, twice: with the anonymization as it is ("masked"), and with the words of the answer
key left as words ("words"). Same model, same prompt: the difference is what over-masking costs.
    progress     each required goal in its range; no goal outside the answer key   (code)
    forbidden    no health detail in anything proposed                             (code)
    notes        each expected note found among the proposed ones                  (judge)

Nothing here writes to the database: the talk is summarized with the pipeline's own functions
(prepare_talk_message, validate_talk), not recorded as a talk.
"""

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from pydantic import BaseModel, ConfigDict, Field

from pacer.ai.anonymization import Anonymizer, PersonKind, Roster, deanonymize
from pacer.ai.answer import AskResult, Coverage, LibraryAnswerer
from pacer.ai.issues import Issue
from pacer.ai.prompts import load_prompt
from pacer.ai.routing import MODELS, ROUTES, AITask, ModelSpec, Router
from pacer.ai.sensitive import remove_sensitive_sentences
from pacer.ai.structured import complete_structured
from pacer.ai.talk_pipeline import PROMPT_VERSION as TALK_PROMPT_VERSION
from pacer.ai.talk_pipeline import TalkExtraction, prepare_talk_message, validate_talk
from pacer.domain import NoteKind, TalkProposalKind

from .judge import Grade, NotesGrade, grade_answer, grade_notes
from .library_set import LibraryCase
from .talk_set import Mention, ResolvedTalk

BASELINE_PROMPT = ("library_baseline", 1)
# The tasks whose model a run compares. The judge stays the same in every run, so runs are comparable.
EVALUATED = (AITask.ANSWER_FROM_LIBRARY, AITask.ANSWER_WITHOUT_LIBRARY, AITask.SUMMARIZE_TALK, AITask.COACH_ADVICE)
_LETTER = r"[^\W\d_]"


def router_for(model: str, models: Mapping[str, ModelSpec] = MODELS) -> Router:
    """The production routes, with the evaluated tasks on `model`. A real Router: the privacy rule and
    the table checks still apply (a model that is not declared, or not allowed, is refused)."""
    routes = {task: replace(route, model=model) if task in EVALUATED else route for task, route in ROUTES.items()}
    return Router(routes=routes, models=models)


class BaselineAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1, max_length=5000, description="Hebrew, 2-6 short paragraphs")


def _total(*costs: float | None) -> float | None:
    """None if any part is unknown: an unknown cost is not zero."""
    return None if any(c is None for c in costs) else sum(c for c in costs if c is not None)


# --- the library --------------------------------------------------------------------------------------


@dataclass
class LibraryCaseResult:
    case: LibraryCase
    answer: AskResult
    retrieved: list[str]  # source titles of the passages, P1 first
    cited: list[str]  # source titles of the passages the answer cited
    seconds: float
    grade: Grade | None = None
    baseline: str | None = None
    baseline_grade: Grade | None = None
    baseline_seconds: float | None = None
    cost_usd: float | None = 0.0

    @property
    def coverage_ok(self) -> bool:
        return self.answer.coverage in self.case.coverage

    @property
    def retrieved_right(self) -> bool | None:
        return any(t in self.case.sources for t in self.retrieved) if self.case.has_source else None

    @property
    def cited_right(self) -> bool | None:
        return any(t in self.case.sources for t in self.cited) if self.case.has_source else None


def run_library_case(
    case: LibraryCase,
    answerer: LibraryAnswerer,
    router: Router,
    anonymizer: Anonymizer,
    titles: dict[int, str],
    *,
    baseline: bool = True,
    judge: bool = True,
) -> LibraryCaseResult:
    started = time.perf_counter()
    answer = answerer.ask(case.question)
    seconds = time.perf_counter() - started
    result = LibraryCaseResult(
        case=case,
        answer=answer,
        retrieved=[titles[c.source_id] for c in answer.citations],
        cited=list(dict.fromkeys(titles[c.source_id] for c in answer.citations if c.cited)),
        seconds=seconds,
        cost_usd=answer.cost_usd,
    )
    costs: list[float | None] = [answer.cost_usd]
    if baseline:
        safe = anonymizer.anonymize(case.question)
        started = time.perf_counter()
        base = complete_structured(
            router.client_for(AITask.ANSWER_WITHOUT_LIBRARY),
            load_prompt(*BASELINE_PROMPT),
            safe.text,
            BaselineAnswer,
            max_tokens=router.route(AITask.ANSWER_WITHOUT_LIBRARY).max_tokens,
        )
        result.baseline_seconds = time.perf_counter() - started
        result.baseline = deanonymize(base.value.answer, safe)
        costs.append(base.cost_usd)
    if judge:
        client = router.client_for(AITask.JUDGE)
        result.grade = grade_answer(
            client,
            anonymizer,
            question=case.question,
            answer=answer.answer,
            checklist=case.judge,
            claims=[(c.text, c.passages) for c in answer.claims],
            passages=answer.passages,
        )
        costs.append(result.grade.cost_usd)
        if result.baseline is not None:
            result.baseline_grade = grade_answer(
                client, anonymizer, question=case.question, answer=result.baseline, checklist=case.judge
            )
            costs.append(result.baseline_grade.cost_usd)
    result.cost_usd = _total(*costs)
    return result


# --- the talks ----------------------------------------------------------------------------------------


def word_spans(notes: str, mentions: Sequence[Mention]) -> list[tuple[int, int]]:
    """Where each mention marked as a WORD is, in `notes`: its n-th appearance as a whole word."""
    spans = []
    for m in mentions:
        if not m.is_word:
            continue
        found = list(re.finditer(rf"(?<!{_LETTER}){re.escape(m.word)}(?!{_LETTER})", notes))
        if len(found) < m.occurrence:
            raise ValueError(f"'{m.word}' #{m.occurrence} is not in the notes as sent")
        spans.append(found[m.occurrence - 1].span())
    return spans


@dataclass
class TalkCaseResult:
    talk: ResolvedTalk
    variant: str  # "masked": anonymization as it is; "words": the answer key's words left as words
    summary: str
    progress: dict[str, int]  # goal title -> the proposed progress
    notes: list[tuple[NoteKind, str]]
    issues: list[Issue]
    seconds: float
    notes_grade: NotesGrade | None = None
    cost_usd: float | None = 0.0
    forbidden_found: list[str] = field(default_factory=list)

    @property
    def required(self) -> list[str]:
        return [p.goal for p in self.talk.case.progress if p.required]

    @property
    def required_in_range(self) -> int:
        expected = {p.goal: p for p in self.talk.case.progress}
        return sum(
            1 for g in self.required if g in self.progress and expected[g].low <= self.progress[g] <= expected[g].high
        )

    @property
    def wrong_progress(self) -> list[str]:
        """Proposals outside the answer key: a goal it does not list, or a value out of its range."""
        expected = {p.goal: p for p in self.talk.case.progress}
        return [
            goal
            for goal, value in self.progress.items()
            if goal not in expected or not expected[goal].low <= value <= expected[goal].high
        ]

    @property
    def notes_found(self) -> int:
        return 0 if self.notes_grade is None else sum(m is not None for m in self.notes_grade.matches)

    @property
    def notes_kind_right(self) -> int:
        if self.notes_grade is None:
            return 0
        expected = self.talk.case.expected_notes
        return sum(
            1
            for e, m in zip(expected, self.notes_grade.matches, strict=True)
            if m is not None and self.notes[m][0] == e.kind
        )


def run_talk_case(
    talk: ResolvedTalk, roster: Roster, router: Router, *, words_stay_words: bool, judge: bool = True
) -> TalkCaseResult:
    case = talk.case
    kept, _ = remove_sensitive_sentences(case.notes)
    spans = word_spans(kept, case.mentions) if words_stay_words else []
    message, _ = prepare_talk_message(
        roster, talk.trainee, talk.counselor, case.held_on, talk.goals, case.notes, words_in_notes=spans
    )
    started = time.perf_counter()
    result = complete_structured(
        router.client_for(AITask.SUMMARIZE_TALK),
        load_prompt("talk_summary", TALK_PROMPT_VERSION),
        message.text,
        TalkExtraction,
        max_tokens=router.route(AITask.SUMMARIZE_TALK).max_tokens,
    )
    seconds = time.perf_counter() - started
    validated = validate_talk(result.value, message, goals=talk.goals)
    titles = {g.id: g.title for g in talk.goals}
    progress = {
        titles[d.goal_id]: d.progress
        for d in validated.drafts
        if d.kind is TalkProposalKind.PROGRESS and d.goal_id is not None and d.progress is not None
    }
    notes = [
        (d.note_kind, d.text) for d in validated.drafts if d.kind is TalkProposalKind.NOTE and d.note_kind is not None
    ]
    everything = " ".join([validated.summary, *(d.text for d in validated.drafts)])
    out = TalkCaseResult(
        talk=talk,
        variant="words" if words_stay_words else "masked",
        summary=validated.summary,
        progress=progress,
        notes=notes,
        issues=validated.issues,
        seconds=seconds,
        forbidden_found=[w for w in case.forbidden if w in everything],
    )
    costs: list[float | None] = [result.cost_usd]
    if judge:
        about = [p for p in (roster.find(PersonKind.TRAINEE, talk.trainee.key),) if p is not None]
        out.notes_grade = grade_notes(
            router.client_for(AITask.JUDGE),
            Anonymizer(roster),
            expected=[(n.kind, n.gist) for n in case.expected_notes],
            proposed=notes,
            prefer=about,
        )
        costs.append(out.notes_grade.cost_usd)
    out.cost_usd = _total(*costs)
    return out


# --- what a run adds up to ------------------------------------------------------------------------------


def _rate(part: int, whole: int) -> float | None:
    return None if whole == 0 else round(part / whole, 3)


def library_summary(results: Sequence[LibraryCaseResult]) -> dict:
    with_source = [r for r in results if r.case.has_source]
    graded = [r for r in results if r.grade is not None]
    base_graded = [r for r in results if r.baseline_grade is not None]
    reviewed = [r for r in results if r.case.reviewed]

    def checklist(grades: Sequence[Grade]) -> float | None:
        return _rate(sum(g.checklist_passed for g in grades), sum(len(g.checklist) for g in grades))

    def helpful(grades: Sequence[Grade]) -> float | None:
        return round(sum(g.helpful for g in grades) / len(grades), 2) if grades else None

    rag_grades = [r.grade for r in graded if r.grade is not None]
    base_grades = [r.baseline_grade for r in base_graded if r.baseline_grade is not None]
    costs = [r.cost_usd for r in results]
    return {
        "cases": len(results),
        "coverage_ok": _rate(sum(r.coverage_ok for r in results), len(results)),
        "coverage_ok_reviewed": _rate(sum(r.coverage_ok for r in reviewed), len(reviewed)),
        "said_none_with_source": sum(r.answer.coverage is Coverage.NONE for r in with_source),
        "answered_without_source": sum(
            r.answer.coverage is not Coverage.NONE for r in results if not r.case.has_source
        ),
        "retrieved_right": _rate(sum(bool(r.retrieved_right) for r in with_source), len(with_source)),
        "cited_right": _rate(sum(bool(r.cited_right) for r in with_source), len(with_source)),
        "validation_issues": sum(len(r.answer.issues) for r in results),
        "claims_supported": _rate(sum(g.claims_supported for g in rag_grades), sum(len(g.claims) for g in rag_grades)),
        "checklist": checklist(rag_grades),
        "checklist_baseline": checklist(base_grades),
        "helpful": helpful(rag_grades),
        "helpful_baseline": helpful(base_grades),
        "seconds_per_answer": round(sum(r.seconds for r in results) / len(results), 2) if results else None,
        "cost_usd": None if any(c is None for c in costs) else round(sum(c for c in costs if c is not None), 4),
    }


def talk_summary(results: Sequence[TalkCaseResult]) -> dict:
    by_variant: dict[str, dict] = {}
    for variant in ("masked", "words"):
        rs = [r for r in results if r.variant == variant]
        if not rs:
            continue
        costs = [r.cost_usd for r in rs]
        expected_notes = sum(len(r.talk.case.expected_notes) for r in rs)
        by_variant[variant] = {
            "cases": len(rs),
            "required_in_range": _rate(sum(r.required_in_range for r in rs), sum(len(r.required) for r in rs)),
            "wrong_progress": sum(len(r.wrong_progress) for r in rs),
            "notes_found": _rate(sum(r.notes_found for r in rs), expected_notes),
            "notes_kind_right": _rate(sum(r.notes_kind_right for r in rs), expected_notes),
            "forbidden_found": sum(len(r.forbidden_found) for r in rs),
            "validation_issues": sum(len(r.issues) for r in rs),
            "seconds_per_talk": round(sum(r.seconds for r in rs) / len(rs), 2),
            "cost_usd": None if any(c is None for c in costs) else round(sum(c for c in costs if c is not None), 4),
        }
    return by_variant
