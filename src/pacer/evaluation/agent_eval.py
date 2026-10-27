"""The agent, evaluated: does it call the right tools, about the right trainee, and answer well?

Per case, in code:
    outcome    answer, referral or not_sent, as the case expects
    tools      every must_call tool called, at least one of each any_of group, no must_not_call tool
    trainee    every call with a trainee_id is about the case's trainee, and at least one is
    text       the answer mentions what it must, and nothing it must not (a fact of another group)
And by the judge, when the case has a checklist: each item passed or not (the items every answer is
checked on, judge.ALWAYS_CHECKED, are reported on their own).

The question this answers is the choice of the agent's model: the same cases, the same tools, the same
coach and gate (they stay on their own routes), and only the model that decides differs.

Nothing here writes to the database. Needs the demo academy, Ollama (bge-m3) and ANTHROPIC_API_KEY.
"""

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pacer.ai.agent import Agent, AgentOutcome, AgentResult, Turn
from pacer.ai.anonymization import Anonymizer
from pacer.ai.llm_client import LLMClient
from pacer.domain import StaffMember
from pacer.repositories import Repos

from .judge import Grade, grade_answer

TRAINEE_TOOLS = frozenset({"trainee_record", "attendance", "ask_coach"})  # tools with a trainee_id


class ChatTurnCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class AgentCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^A\d{2,}$")
    tests: str
    asker: str  # a staff member's full name
    history: list[ChatTurnCase] = Field(default_factory=list)
    message: str = Field(min_length=5)
    outcome: Literal["answer", "referral", "not_sent"]
    must_call: list[str] = Field(default_factory=list)
    any_of: list[list[str]] = Field(default_factory=list)  # at least one tool of each group
    must_not_call: list[str] = Field(default_factory=list)
    about: list[str] = Field(default_factory=list)  # trainees (full names) the calls must be about
    answer_mentions: list[str] = Field(default_factory=list)
    answer_must_not: list[str] = Field(default_factory=list)
    checklist: list[str] = Field(default_factory=list)  # for the judge


class AgentEvalSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: int = Field(ge=1)
    about: str
    cases: list[AgentCase] = Field(min_length=1)

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"


def load_agent_set(path: Path) -> AgentEvalSet:
    return AgentEvalSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class ResolvedAgentCase:
    case: AgentCase
    asker_id: int
    about_ids: frozenset[int]


def resolve_agent_cases(
    eval_set: AgentEvalSet, repos: Repos, tools: Sequence[str]
) -> tuple[list[ResolvedAgentCase], list[str]]:
    """Names to ids, and every problem the set has (a typo in a name or a tool would silently test nothing)."""
    staff = {s.full_name: s.key for s in repos.staff.list_all()}
    trainees = {t.full_name: t.key for t in repos.trainees.list_all()}
    known_tools = set(tools)
    resolved, problems = [], []
    for case in eval_set.cases:
        named = [*case.must_call, *case.must_not_call, *(t for group in case.any_of for t in group)]
        problems += [f"{case.id}: no tool named {t}" for t in named if t not in known_tools]
        problems += [f"{case.id}: no trainee named {name}" for name in case.about if name not in trainees]
        if case.asker not in staff:
            problems.append(f"{case.id}: no staff member named {case.asker}")
            continue
        about = frozenset(trainees[n] for n in case.about if n in trainees)
        resolved.append(ResolvedAgentCase(case, staff[case.asker], about))
    return resolved, problems


# --- one case --------------------------------------------------------------------------------------


@dataclass
class AgentCaseResult:
    resolved: ResolvedAgentCase
    result: AgentResult
    seconds: float
    grade: Grade | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def case(self) -> AgentCase:
        return self.resolved.case

    @property
    def called(self) -> list[str]:
        return [step.tool for step in self.result.steps]

    @property
    def outcome_ok(self) -> bool:
        return self.result.outcome.value == self.case.outcome

    @property
    def tools_ok(self) -> bool:
        called = set(self.called)
        return (
            set(self.case.must_call) <= called
            and all(called & set(group) for group in self.case.any_of)
            and not called & set(self.case.must_not_call)
        )

    @property
    def trainee_ok(self) -> bool | None:
        """None when the case names no trainee. Otherwise: no call about anyone else, and at least one about
        them (a case that names a trainee expects the agent to look them up)."""
        if not self.resolved.about_ids:
            return None
        # Successful calls only: their arguments were validated (an int, even if the model wrote "15"),
        # and a call that came back "not found" was about no one (review, 2026-10-27).
        ids = [
            step.arguments.get("trainee_id")
            for step in self.result.steps
            if step.ok and step.tool in TRAINEE_TOOLS and step.arguments.get("trainee_id") is not None
        ]
        return bool(ids) and all(i in self.resolved.about_ids for i in ids)

    @property
    def text_ok(self) -> bool | None:
        if not self.case.answer_mentions and not self.case.answer_must_not:
            return None
        text = self.result.answer or ""
        return all(m in text for m in self.case.answer_mentions) and not any(
            m in text for m in self.case.answer_must_not
        )

    @property
    def passed(self) -> bool:
        """Every check made in code. The judge's checklist is reported apart: it is a judgement."""
        return self.outcome_ok and self.tools_ok and self.trainee_ok is not False and self.text_ok is not False


def run_agent_case(
    resolved: ResolvedAgentCase,
    agent: Agent,
    actors: dict[int, StaffMember],
    judge: LLMClient | None,
    anonymizer: Anonymizer,
) -> AgentCaseResult:
    case = resolved.case
    started = time.perf_counter()
    history = [Turn(t.role, t.content) for t in case.history]
    result = agent.chat(actors[resolved.asker_id], case.message, history)
    out = AgentCaseResult(resolved, result, time.perf_counter() - started)
    if judge is None or not case.checklist or result.outcome is not AgentOutcome.ANSWER or not result.answer:
        return out
    try:  # the JUDGE failing is not the agent failing: its answer and the code checks are kept
        out.grade = grade_answer(
            judge, anonymizer, question=case.message, answer=result.answer, checklist=case.checklist
        )
    except Exception as error:  # noqa: BLE001 - reported in the case, the run goes on
        out.notes.append(f"judge failed: {type(error).__name__}")
    return out


# --- the summary and the report ------------------------------------------------------------------


def _rate(values: Sequence[bool | None]) -> float | None:
    counted = [v for v in values if v is not None]
    return None if not counted else round(sum(counted) / len(counted), 3)


def agent_summary(results: Sequence[AgentCaseResult]) -> dict:
    checklist: list[bool | None] = []
    always: list[bool | None] = []
    for r in results:
        if r.grade is not None:
            own = len(r.case.checklist)
            checklist += r.grade.checklist[:own]
            always += r.grade.checklist[own:]
    answered = [r for r in results if r.result.outcome is AgentOutcome.ANSWER]
    costs = [r.result.cost_usd for r in results]
    judge_costs = [r.grade.cost_usd for r in results if r.grade is not None]
    return {
        "cases": len(results),
        "passed": _rate([r.passed for r in results]),
        "outcome_ok": _rate([r.outcome_ok for r in results]),
        "tools_ok": _rate([r.tools_ok for r in results]),
        "trainee_ok": _rate([r.trainee_ok for r in results]),
        "text_ok": _rate([r.text_ok for r in results]),
        "checklist": _rate(checklist),
        "always_checked": _rate(always),
        "tool_calls_per_answer": round(sum(len(r.result.steps) for r in answered) / len(answered), 2)
        if answered
        else None,
        "failed_tool_calls": sum(not s.ok for r in results for s in r.result.steps),
        "stopped": sum(r.result.outcome is AgentOutcome.STOPPED for r in results),
        "seconds_per_case": round(sum(r.seconds for r in results) / len(results), 1) if results else None,
        "cost_usd": None if any(c is None for c in costs) else round(sum(c for c in costs if c is not None), 4),
        "judge_cost_usd": None if any(c is None for c in judge_costs) else round(sum(c for c in judge_costs if c), 4),
    }


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{round(value * 100)}%"


def _num(value: float | None) -> str:
    return "-" if value is None else str(value)


def _mark(value: bool | None) -> str:
    return "-" if value is None else ("✓" if value else "✗")


def render_agent(results: Sequence[AgentCaseResult], summary: dict, meta: Sequence[str]) -> str:
    lines = [
        "# הערכת הסוכן",
        "",
        *[f"- {m}" for m in meta],
        "",
        "| | |",
        "|---|---|",
        f"| מקרים שעברו את כל הבדיקות בקוד | {_pct(summary['passed'])} |",
        f"| תוצאה נכונה (תשובה / הפניה / לא נשלח) | {_pct(summary['outcome_ok'])} |",
        f"| בחירת tools נכונה | {_pct(summary['tools_ok'])} |",
        f"| הקריאות על החניך הנכון | {_pct(summary['trainee_ok'])} |",
        f"| התשובה מזכירה מה שצריך, ולא מה שאסור | {_pct(summary['text_ok'])} |",
        f"| סעיפי בדיקה (שופט) | {_pct(summary['checklist'])} |",
        f"| בלי אבחנה, תוויות רפואיות או עצה טיפולית (שופט) | {_pct(summary['always_checked'])} |",
        f"| קריאות tools לתשובה | {_num(summary['tool_calls_per_answer'])} |",
        f"| קריאות שנכשלו (ארגומנט שגוי, לא נמצא) | {summary['failed_tool_calls']} |",
        f"| נעצרו בגבול | {summary['stopped']} |",
        f"| שניות למקרה | {summary['seconds_per_case']} |",
        f"| עלות: סוכן (כולל המלווה והשער) ${summary['cost_usd']}, שופט ${summary['judge_cost_usd']} | |",
    ]
    for r in results:
        case = r.case
        lines += ["", f"## {case.id}: {case.tests}", ""]
        for turn in case.history:
            lines.append(f"> ({turn.role}) {turn.content}")
        lines += [f"**{case.asker}:** {case.message}", ""]
        lines.append(
            f"תוצאה: {r.result.outcome} (צפוי {case.outcome}) {_mark(r.outcome_ok)} · tools {_mark(r.tools_ok)}"
            f" · חניך {_mark(r.trainee_ok)} · טקסט {_mark(r.text_ok)} · {r.seconds:.0f} שניות · {r.result.rounds} סבבים"
        )
        expected = []
        if case.must_call:
            expected.append("חייב: " + ", ".join(case.must_call))
        expected += ["אחד מ: " + " / ".join(group) for group in case.any_of]
        if case.must_not_call:
            expected.append("אסור: " + ", ".join(case.must_not_call))
        if expected:
            lines.append("צפוי: " + " · ".join(expected))
        lines.append("")
        for step in r.result.steps:
            args = json.dumps(step.arguments, ensure_ascii=False)
            lines.append(f"1. `{step.tool}` {args} → {step.summary}" + ("" if step.ok else " ✗"))
        if r.result.answer:
            lines += ["", "```", r.result.answer, "```"]
        elif r.result.referral is not None:
            lines += ["", "(הנוהל לאירוע חריג)"]
        if r.grade:
            lines += ["", "| סעיף | |", "|---|---|"]
            for item, verdict in zip(case.checklist, r.grade.checklist, strict=False):
                lines.append(f"| {item} | {_mark(verdict)} |")
        if r.notes:
            lines += ["", "הערות: " + "; ".join(r.notes)]
        if r.result.issues:
            lines += ["", "בעיות: " + "; ".join(f"{i.code}: {i.detail}" for i in r.result.issues)]
    return "\n".join(lines)
