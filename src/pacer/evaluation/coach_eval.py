"""The counselor coach, evaluated: does it advise well, and does knowing the trainee help?

Per case:
    outcome      advice or referral, as the case expects                                    (code)
    checklist    the case's "must / must not" items, plus the line the system always holds  (judge)
    claims       each GROUNDED step: does the passage it cites really support it?          (judge)
    helpful      1 to 5                                                                     (judge)
    grounded     how many steps rest on a library passage                                   (code)
    baseline     the same question WITHOUT the trainee's record (a general question): the same checklist.
                 The difference is what knowing the trainee adds. Most items are about this trainee
                 ("builds on what worked with him"), so a general answer is expected to miss them: that
                 gap is the point of the measurement.

Nothing here writes to the database. Needs the demo academy, Ollama (bge-m3) and ANTHROPIC_API_KEY.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pacer.ai.anonymization import Anonymizer
from pacer.ai.coach import Coach, CoachOutcome, CoachResult
from pacer.ai.llm_client import LLMClient
from pacer.domain import StaffMember
from pacer.repositories import Repos

from .judge import Grade, grade_answer


class CoachCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^C\d{2,}$")
    tests: str
    trainee: str | None  # a demo trainee's full name; None for a general question
    asker: str  # the staff member who asks: must be allowed to see the trainee
    question: str = Field(min_length=5)
    outcome: Literal["advice", "referral"]
    checklist: list[str]

    @model_validator(mode="after")
    def advice_has_a_checklist(self) -> Self:
        if (self.outcome == "advice") != bool(self.checklist):
            raise ValueError(f"{self.id}: an advice case needs a checklist; a referral case has none")
        return self


class CoachEvalSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: int = Field(ge=1)
    about: str
    cases: list[CoachCase] = Field(min_length=1)

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"


def load_coach_set(path: Path) -> CoachEvalSet:
    return CoachEvalSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ResolvedCase:
    case: CoachCase
    asker_id: int
    trainee_id: int | None


def resolve_cases(eval_set: CoachEvalSet, repos: Repos) -> tuple[list[ResolvedCase], list[str]]:
    """Names to ids, and every problem the set has with this academy (a typo would silently test nothing)."""
    staff = {s.full_name: s for s in repos.staff.list_all()}
    trainees = {t.full_name: t for t in repos.trainees.list_all()}
    resolved, problems = [], []
    for case in eval_set.cases:
        asker = staff.get(case.asker)
        trainee = trainees.get(case.trainee) if case.trainee else None
        if asker is None:
            problems.append(f"{case.id}: no staff member named {case.asker}")
        if case.trainee and trainee is None:
            problems.append(f"{case.id}: no trainee named {case.trainee}")
        if asker is not None and (case.trainee is None or trainee is not None):
            resolved.append(ResolvedCase(case, asker.key, trainee.key if trainee else None))
    return resolved, problems


def advice_text(result: CoachResult) -> str:
    """The answer as a counselor reads it, for the judge and the report. Each step says its grounding."""
    if result.outcome is CoachOutcome.REFERRAL and result.referral is not None:
        return result.referral.message + "\n" + "\n".join(f"- {s.action}" for s in result.referral.steps)
    if result.advice is None:
        return result.note
    a = result.advice
    lines = [a.understanding, ""]
    for number, step in enumerate(a.steps, 1):
        basis = f"[מבוסס מקור: {', '.join(step.passages)}]" if step.grounded else "[המלצה כללית]"
        lines.append(f"{number}. {step.action} {basis}")
        if step.example:
            lines.append(f'   לדוגמה: "{step.example}"')
    if a.questions:
        lines += ["", "שאלות הבהרה:", *[f"- {q}" for q in a.questions]]
    if a.involve:
        lines += ["", "את מי לערב:", *[f"- {role.label}: {why}" for role, why in a.involve]]
    return "\n".join(lines)


@dataclass
class CoachCaseResult:
    case: CoachCase
    result: CoachResult
    seconds: float
    grade: Grade | None = None
    baseline: CoachResult | None = None
    baseline_grade: Grade | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def outcome_ok(self) -> bool:
        return self.result.outcome.value == self.case.outcome

    @property
    def steps(self) -> int:
        return len(self.result.advice.steps) if self.result.advice else 0

    @property
    def grounded(self) -> int:
        return sum(s.grounded for s in self.result.advice.steps) if self.result.advice else 0


def run_coach_case(
    resolved: ResolvedCase,
    coach: Coach,
    actors: dict[int, StaffMember],
    judge: LLMClient | None,
    anonymizer: Anonymizer,
    *,
    baseline: bool,
) -> CoachCaseResult:
    case = resolved.case
    actor = actors[resolved.asker_id]
    started = time.perf_counter()
    result = coach.ask(actor, case.question, resolved.trainee_id)
    seconds = time.perf_counter() - started
    out = CoachCaseResult(case, result, seconds)
    if result.advice is None or judge is None:
        return out
    claims = [(s.action, s.passages) for s in result.advice.steps if s.grounded]
    out.grade = grade_answer(
        judge,
        anonymizer,
        question=case.question,
        answer=advice_text(result),
        checklist=case.checklist,
        claims=claims,
        passages=result.passages,
    )
    if baseline and resolved.trainee_id is not None:
        # The same question about the same trainee (same permissions, same reading of names), without
        # the record: the only difference between the two answers is what the coach knows about them.
        # A failure here is noted and the case is kept: its main answer and grade are already paid for.
        try:
            general = coach.ask(actor, case.question, resolved.trainee_id, use_record=False)
            out.baseline = general
            if general.advice is not None:
                out.baseline_grade = grade_answer(
                    judge,
                    anonymizer,
                    question=case.question,
                    answer=advice_text(general),
                    checklist=case.checklist,
                    claims=[(s.action, s.passages) for s in general.advice.steps if s.grounded],
                    passages=general.passages,
                )
        except Exception as error:  # noqa: BLE001 - reported in the case, the run goes on
            out.notes.append(f"baseline failed: {type(error).__name__}")
    return out


def _rate(part: int, whole: int) -> float | None:
    return None if whole == 0 else round(part / whole, 3)


def _checklist(results: Sequence[CoachCaseResult], *, baseline: bool = False) -> tuple[float | None, int]:
    """The share of the CASE's own checklist items that passed, and how many verdicts the judge skipped.
    Each grade ends with the items every answer is checked on (judge.ALWAYS_CHECKED): they are not part
    of the case's checklist, so they are left out here and reported on their own (always_checked).
    A skipped verdict (None) is neither a pass nor a fail."""
    verdicts: list[bool | None] = []
    for r in results:
        grade = r.baseline_grade if baseline else r.grade
        if grade is not None:
            verdicts += grade.checklist[: len(r.case.checklist)]
    known = [v for v in verdicts if v is not None]
    return _rate(sum(known), len(known)), len(verdicts) - len(known)


def _always_checked(results: Sequence[CoachCaseResult]) -> float | None:
    known = [v for r in results if r.grade for v in r.grade.checklist[len(r.case.checklist) :] if v is not None]
    return _rate(sum(known), len(known))


def coach_summary(results: Sequence[CoachCaseResult]) -> dict:
    advised = [r for r in results if r.case.outcome == "advice"]
    with_trainee = [r for r in advised if r.case.trainee is not None and r.baseline_grade is not None]
    claims = [v for r in advised if r.grade for v in r.grade.claims if v is not None]
    helpful = [r.grade.helpful for r in advised if r.grade]
    costs = [r.result.cost_usd for r in results] + [r.baseline.cost_usd for r in results if r.baseline]
    judge_costs = [g.cost_usd for r in results for g in (r.grade, r.baseline_grade) if g]
    return {
        "cases": len(results),
        "outcome_ok": _rate(sum(r.outcome_ok for r in results), len(results)),
        "checklist": _checklist(advised)[0],
        # personal cases only (those with a baseline), with and without the record, on the same items
        "checklist_personal": _checklist(with_trainee)[0],
        "checklist_personal_without_record": _checklist(with_trainee, baseline=True)[0],
        "always_checked": _always_checked(advised),  # no diagnosis, no medical labels, no treatment advice
        "skipped_verdicts": _checklist(advised)[1] + _checklist(with_trainee, baseline=True)[1],
        "helpful": None if not helpful else round(sum(helpful) / len(helpful), 2),
        "grounded_steps": _rate(sum(r.grounded for r in advised), sum(r.steps for r in advised)),
        "claims_supported": _rate(sum(v is True for v in claims), len(claims)),
        "issues": sum(len(r.result.advice.issues) for r in advised if r.result.advice),
        "seconds_per_answer": None if not results else round(sum(r.seconds for r in results) / len(results), 1),
        "cost_usd": None if any(c is None for c in costs) else round(sum(c for c in costs if c is not None), 4),
        "judge_cost_usd": None
        if any(c is None for c in judge_costs)
        else round(sum(c for c in judge_costs if c is not None), 4),
    }


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{round(value * 100)}%"


def render_coach(results: Sequence[CoachCaseResult], summary: dict, meta: Sequence[str]) -> str:
    lines = [
        "# הערכת המלווה",
        "",
        *[f"- {m}" for m in meta],
        "",
        "| | |",
        "|---|---|",
        f"| תוצאה נכונה (עצה / הפניה) | {_pct(summary['outcome_ok'])} |",
        f"| סעיפי בדיקה שעברו | {_pct(summary['checklist'])} |",
        f"| סעיפים אישיים, **עם** הרשומה של החניך | {_pct(summary['checklist_personal'])} |",
        f"| אותם סעיפים, **בלי** הרשומה | {_pct(summary['checklist_personal_without_record'])} |",
        f"| בלי אבחנה, תוויות רפואיות או עצה טיפולית | {_pct(summary['always_checked'])} |",
        f"| מועילות (שופט, 1-5) | {summary['helpful'] or '-'} |",
        f"| צעדים מבוססי מקור | {_pct(summary['grounded_steps'])} |",
        f"| ציטוטים שהקטע באמת תומך בהם | {_pct(summary['claims_supported'])} |",
        f"| בעיות שהוולידציה תיקנה | {summary['issues']} |",
        f"| שניות לתשובה | {summary['seconds_per_answer']} |",
        f"| עלות: מלווה ${summary['cost_usd']}, שופט ${summary['judge_cost_usd']} | |",
    ]
    for r in results:
        lines += ["", f"## {r.case.id}: {r.case.tests}", "", f"**שאלה:** {r.case.question}", ""]
        mark = "✓" if r.outcome_ok else "✗"
        lines.append(f"תוצאה: {r.result.outcome} (צפוי {r.case.outcome}) {mark} · {r.seconds:.0f} שניות")
        lines += ["", "```", advice_text(r.result), "```"]
        if r.grade:
            items = [*r.case.checklist]
            verdicts = r.grade.checklist[: len(items)]
            lines += ["", "| סעיף | עם הרשומה | בלי |", "|---|---|---|"]
            for i, item in enumerate(items):
                without = r.baseline_grade.checklist[i] if r.baseline_grade else None
                lines.append(f"| {item} | {_mark(verdicts[i])} | {_mark(without)} |")
            lines += ["", f"מועילות: {r.grade.helpful}/5. {r.grade.why}"]
        if r.notes:
            lines += ["", "הערות: " + "; ".join(r.notes)]
        if r.result.advice and r.result.advice.issues:
            issues = "; ".join(f"{i.code} ({i.section} {i.item}): {i.detail}" for i in r.result.advice.issues)
            lines += ["", f"בעיות: {issues}"]
    return "\n".join(lines)


def _mark(value: bool | None) -> str:
    return "-" if value is None else ("✓" if value else "✗")
