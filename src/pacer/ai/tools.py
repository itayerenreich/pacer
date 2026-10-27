"""The agent's tools: thin wrappers over services Pacer already has (stage 6).

A tool is a name, a description, a Pydantic model of its input, and a function. The model sees the
name, the description and the JSON Schema that Pydantic generates; it can only ASK for a tool. Our
code runs it, and everything that protects the data is on our side:

* The actor is bound in code. No tool has a "who is asking" parameter, so the model cannot ask in
  someone else's name. A trainee the asker may not see is "not found", exactly like the API's 404.
* Arguments are validated by Pydantic (extra="forbid"). A wrong argument is an error RESULT that goes
  back to the model, which can correct itself; it never crashes the conversation.
* What a tool returns is plain text WITH names, built here. The agent removes health sentences and
  anonymizes it before the model reads it (ai/agent.py), so a tool cannot forget to.
* Read only (decision 2026-10-27): no tool changes anything. The personal file is not a tool, ever.

Each tool returns one item per line, so that a line about health is removed on its own.
"""

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from pacer.domain import Cohort, DomainRuleError, StaffMember, Trainee, TraineeStatus, Weekday, week_start
from pacer.repositories import UnitOfWork
from pacer.services import (
    AttendanceService,
    AttendanceSummary,
    CoachContextService,
    NotFoundError,
    PermissionDeniedError,
    RuleViolationError,
    ScheduleService,
    TraineeService,
)

from .anonymization import AnonymizationSession, Person, PersonKind, deanonymize
from .coach import Coach, CoachOutcome, trainee_lines
from .embeddings import Embedder
from .library import retrieve
from .llm_client import LLMError
from .safety import REFERRAL, Referral

log = logging.getLogger(__name__)

MAX_RESULT_CHARS = 6000  # a tool's result is cut here: the model reads summaries, not databases
MAX_TRAINEES_LISTED = 40


@dataclass(frozen=True)
class SourceRef:
    """A library source a tool showed the model: listed under the answer, so the counselor can check."""

    source_id: int
    citation: str
    evidence: str
    section: str | None = None


@dataclass
class ToolOutput:
    text: str  # with names: anonymized by the agent before any model reads it
    summary: str  # for the trace the counselor sees: counts and ids only, never names or content
    about: list[Person] = field(default_factory=list)  # who it is about: an ambiguous first name is them
    sources: list[SourceRef] = field(default_factory=list)
    stop: Referral | None = None  # an exceptional incident: the conversation ends with the procedure
    cost_usd: float | None = 0.0  # models the tool itself called (the coach); None when unknown


@dataclass(frozen=True)
class ToolContext:
    """What every tool run gets. The actor comes from the request, never from the model."""

    actor: StaffMember
    session: AnonymizationSession
    today: date


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: type[BaseModel]
    run: Callable[[ToolContext, Any], ToolOutput]

    def schema(self) -> dict[str, Any]:
        """The function as the model sees it (OpenAI format; LiteLLM translates it for Claude)."""
        parameters = self.params.model_json_schema()
        parameters.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": parameters},
        }


class ToolError(Exception):
    """A tool could not run. Its message goes back to the model as the result: no names, no data."""


@dataclass(frozen=True)
class ToolRun:
    """One tool call as it happened: what the agent sends back to the model, and what the trace shows."""

    name: str
    arguments: dict[str, Any]  # as the model wrote them (tokens, not names)
    ok: bool
    output: ToolOutput
    latency_ms: int


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        names = [t.name for t in tools]
        if len(set(names)) != len(names):
            raise ValueError(f"Two tools with one name: {names}")
        self._tools = {t.name: t for t in tools}

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def run(self, context: ToolContext, name: str, arguments: str) -> ToolRun:
        """Never raises for anything the model did: an unknown tool, broken JSON, a wrong argument, a
        trainee it may not see. All of them are results the model reads and can react to."""
        started = time.perf_counter()
        parsed: dict[str, Any] = {}
        try:
            tool = self._tools.get(name)
            if tool is None:
                raise ToolError(f"There is no tool named {name!r}. The tools are: {', '.join(self._tools)}")
            try:
                loaded = json.loads(arguments or "{}")
            except json.JSONDecodeError as error:
                raise ToolError(f"The arguments are not valid JSON: {error.msg}") from error
            if not isinstance(loaded, dict):
                raise ToolError("The arguments must be a JSON object")
            parsed = loaded
            try:
                params = tool.params.model_validate(_restore_names(parsed, context.session))
            except ValidationError as error:
                raise ToolError(_validation_message(error)) from error
            output, ok = tool.run(context, params), True
        except ToolError as error:
            output, ok = ToolOutput(f"Error: {error}", "error"), False
        except NotFoundError:
            # The same words whether the record does not exist or is not this actor's: nothing leaks.
            output, ok = ToolOutput("Error: not found (it does not exist, or is not yours to see)", "not found"), False
        except (PermissionDeniedError, RuleViolationError, DomainRuleError) as error:
            output, ok = ToolOutput(f"Error: {error}", "refused"), False
        except LLMError as error:
            log.warning("Tool %s: a model it needs failed: %s", name, type(error).__name__)
            output, ok = (
                ToolOutput("Error: this tool is unavailable right now. Answer without it.", "unavailable"),
                False,
            )
        log.info("Tool %s: ok=%s by staff %s", name, ok, context.actor.id)  # never the arguments or the result
        return ToolRun(name, parsed, ok, output, round((time.perf_counter() - started) * 1000))


def _restore_names(value: Any, session: AnonymizationSession) -> Any:
    """The model writes tokens ("[TRAINEE_7]", "[NAME_1]"); the tools run on this computer, with names."""
    if isinstance(value, str):
        return deanonymize(value, session.view())
    if isinstance(value, list):
        return [_restore_names(v, session) for v in value]
    if isinstance(value, dict):
        return {k: _restore_names(v, session) for k, v in value.items()}
    return value


def _validation_message(error: ValidationError) -> str:
    """Which argument is wrong and why, WITHOUT the value (it may be a name)."""
    problems = [f"{'.'.join(str(p) for p in e['loc']) or 'arguments'}: {e['msg']}" for e in error.errors()]
    return "Invalid arguments: " + "; ".join(problems)


def _cut(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return text[:MAX_RESULT_CHARS].rsplit("\n", 1)[0] + "\n(cut: the result was too long)"


# --- the inputs ----------------------------------------------------------------------------------


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


TRAINEE_ID = Field(description="The number inside the trainee's token: [TRAINEE_7] -> 7", ge=1)


class FindTrainees(_Params):
    name: str | None = Field(
        default=None,
        max_length=100,
        description="A token as written in the conversation ([TRAINEE_7] or [NAME_1]), or omit to list every "
        "trainee you may see",
    )


class TraineeRecord(_Params):
    trainee_id: int = TRAINEE_ID


class Attendance(_Params):
    trainee_id: int | None = Field(default=None, ge=1, description="One trainee; omit for a whole group or everyone")
    group_id: int | None = Field(default=None, ge=1, description="One group; omit for every trainee you may see")
    days: int = Field(default=30, ge=1, le=120, description="How many days back from today")


class WeekSchedule(_Params):
    day: date | None = Field(default=None, description="Any day of the week wanted (YYYY-MM-DD); omit for this week")


class SearchLibrary(_Params):
    query: str = Field(min_length=3, max_length=500, description="What to look for, in Hebrew, in a few words")


class AskCoach(_Params):
    question: str = Field(min_length=5, max_length=2000, description="The counselor's question, in Hebrew")
    trainee_id: int | None = Field(
        default=None, ge=1, description="The trainee it is about; omit for a general question"
    )


class ReportIncident(_Params):
    pass


# --- the tools -----------------------------------------------------------------------------------


class AgentTools:
    """The tools, over the same services and permission rules as the API. Each run opens its own short
    transaction: none is open while a model works, here or in the coach."""

    def __init__(self, uow: UnitOfWork, coach: Coach, embedder: Embedder, chunking: str, *, k: int = 4) -> None:
        self._uow = uow
        self._coach = coach
        self._embedder = embedder
        self._chunking = chunking
        self._k = k

    def registry(self) -> ToolRegistry:
        return ToolRegistry(
            [
                Tool(
                    "find_trainees",
                    "Find trainees by a name token from the conversation, or list every trainee the counselor may "
                    "see. Returns each trainee's token, id and group. Use it to know who [NAME_n] is.",
                    FindTrainees,
                    self.find_trainees,
                ),
                Tool(
                    "trainee_record",
                    "What the academy knows about one trainee: goals with progress and recent updates, notes "
                    "approved by counselors (what worked, what did not), and attendance of the last 30 days.",
                    TraineeRecord,
                    self.trainee_record,
                ),
                Tool(
                    "attendance",
                    "Attendance rates: for one trainee, one group, or every trainee the counselor may see (lowest "
                    "first), with the sessions missed most. Rate = present / (present + absent); excused not counted.",
                    Attendance,
                    self.attendance,
                ),
                Tool(
                    "week_schedule",
                    "The academy's schedule for one week, Sunday to Saturday, with cancelled sessions marked.",
                    WeekSchedule,
                    self.week_schedule,
                ),
                Tool(
                    "search_library",
                    "Search the academy's professional library (education, research, tradition) by meaning. Returns "
                    "passages with their source and kind of evidence. Use it before giving advice.",
                    SearchLibrary,
                    self.search_library,
                ),
                Tool(
                    "ask_coach",
                    "Ask the counselor coach for practical educational advice. With a trainee_id it reads the "
                    "trainee's record itself. Returns steps, each marked as resting on a library source or as "
                    "general advice. Use it for 'how should I deal with...' questions.",
                    AskCoach,
                    self.ask_coach,
                ),
                Tool(
                    "report_exceptional_incident",
                    "Call this INSTEAD of answering if anything suggests someone may be at risk: self-harm, "
                    "violence, abuse, or harm to a trainee. The academy's fixed procedure is then shown.",
                    ReportIncident,
                    self.report_incident,
                ),
            ]
        )

    # -- trainees ---------------------------------------------------------------------------------

    def find_trainees(self, ctx: ToolContext, p: FindTrainees) -> ToolOutput:
        with self._uow() as repos:
            service = TraineeService(repos.trainees, repos.groups, repos.personal_files, repos.approvals)
            trainees = service.list(ctx.actor)
            groups = {g.key: g.name for c in repos.cohorts.list_all() for g in repos.groups.list_by_cohort(c.key)}
        wanted = (p.name or "").strip()
        if wanted:
            trainees = [t for t in trainees if _matches(t, wanted)]
        trainees.sort(key=lambda t: (t.status is not TraineeStatus.ACTIVE, t.key))
        if not trainees:
            return ToolOutput("No trainee you may see matches.", "0 trainees")
        lines = [f"{len(trainees)} trainee(s):"]
        for t in trainees[:MAX_TRAINEES_LISTED]:
            lines.append(
                f"- {t.full_name} (trainee_id {t.key}) · group {groups.get(t.group_id, t.group_id)} "
                f"(group_id {t.group_id}) · {t.status}"
            )
        if len(trainees) > MAX_TRAINEES_LISTED:
            lines.append(f"(and {len(trainees) - MAX_TRAINEES_LISTED} more)")
        about = [_person(t) for t in trainees[:MAX_TRAINEES_LISTED]]
        return ToolOutput("\n".join(lines), f"{len(trainees)} trainees", about=about)

    def trainee_record(self, ctx: ToolContext, p: TraineeRecord) -> ToolOutput:
        with self._uow() as repos:
            context = CoachContextService(
                repos.trainees,
                repos.groups,
                repos.goals,
                repos.goal_updates,
                repos.notes,
                repos.sessions,
                repos.attendance,
            ).build(ctx.actor, p.trainee_id, today=ctx.today)
        lines = [f"trainee_id {context.trainee.key}", *trainee_lines(context)]
        summary = f"trainee {p.trainee_id}: {len(context.goals)} goals, {len(context.notes)} notes"
        return ToolOutput("\n".join(lines), summary, about=[_person(context.trainee)])

    def attendance(self, ctx: ToolContext, p: Attendance) -> ToolOutput:
        if p.trainee_id is not None and p.group_id is not None:
            raise ToolError("Give trainee_id or group_id, not both")
        first = ctx.today - timedelta(days=p.days - 1)
        with self._uow() as repos:
            service = AttendanceService(repos.sessions, repos.attendance, repos.trainees, repos.groups)
            if p.trainee_id is not None:
                summaries = [service.trainee_summary(ctx.actor, p.trainee_id, first, ctx.today, today=ctx.today)]
            else:
                if p.group_id is not None:
                    group = repos.groups.get(p.group_id)
                    if group is None:
                        raise NotFoundError(f"Group {p.group_id} not found")
                    cohorts = [group.cohort_id]
                else:
                    cohorts = [c.key for c in _current_cohorts(repos.cohorts.list_all(), ctx.today)]
                summaries = [
                    s
                    for cohort_id in cohorts
                    for s in service.group_summary(
                        ctx.actor, cohort_id, first, ctx.today, today=ctx.today, group_id=p.group_id
                    )
                ]
        if not summaries:
            return ToolOutput("No trainee you may see.", "0 trainees")
        lines = [f"Attendance from {first} to {ctx.today}, lowest rate first:"]
        lines += [line for s in summaries[:MAX_TRAINEES_LISTED] for line in _attendance_lines(s)]
        if len(summaries) > MAX_TRAINEES_LISTED:
            lines.append(f"(and {len(summaries) - MAX_TRAINEES_LISTED} more)")
        about = [_person(s.trainee) for s in summaries[:MAX_TRAINEES_LISTED]]
        return ToolOutput("\n".join(lines), f"{len(summaries)} trainees, {p.days} days", about=about)

    # -- the schedule -----------------------------------------------------------------------------

    def week_schedule(self, ctx: ToolContext, p: WeekSchedule) -> ToolOutput:
        sunday = week_start(p.day or ctx.today)
        with self._uow() as repos:
            service = ScheduleService(repos.slots, repos.sessions, repos.cohorts, repos.groups, repos.attendance)
            cohorts = _current_cohorts(repos.cohorts.list_all(), sunday)
            sessions = [s for c in cohorts for s in service.week(c.key, sunday)]
            groups = {g.key: g.name for c in cohorts for g in repos.groups.list_by_cohort(c.key)}
        if not sessions:
            return ToolOutput(f"Nothing is published for the week of {sunday}.", f"week of {sunday}: empty")
        lines = [f"The week of Sunday {sunday}:"]
        for s in sorted(sessions, key=lambda s: (s.on, s.start, s.key)):
            who = f" · group {groups.get(s.group_id, s.group_id)}" if s.group_id is not None else ""
            lines.append(
                f"- {Weekday(s.on.weekday()).label} {s.on} {s.start:%H:%M}-{s.end:%H:%M} · {s.title}{who}"
                + (" · CANCELLED" if s.cancelled else "")
            )
        return ToolOutput("\n".join(lines), f"week of {sunday}: {len(sessions)} sessions")

    # -- the library and the coach ----------------------------------------------------------------

    def search_library(self, ctx: ToolContext, p: SearchLibrary) -> ToolOutput:
        hits = retrieve(self._uow, self._embedder, self._chunking, p.query, k=self._k, for_cloud_model=True)
        if not hits:
            return ToolOutput("No passage in the library is close to this.", "0 passages")
        lines: list[str] = []
        sources: list[SourceRef] = []
        for hit in hits:
            where = " · ".join(x for x in (hit.chunk.section, hit.chunk.pages) if x)
            lines += [
                f"[{hit.source.citation} · {hit.source.evidence_level.label}" + (f" · {where}]" if where else "]"),
                " ".join(hit.chunk.text.split()),  # one line: removed whole if it is about health
            ]
            sources.append(
                SourceRef(hit.source.key, hit.source.citation, hit.source.evidence_level.label, hit.chunk.section)
            )
        return ToolOutput("\n".join(lines), f"{len(hits)} passages", sources=sources)

    def ask_coach(self, ctx: ToolContext, p: AskCoach) -> ToolOutput:
        result = self._coach.ask(ctx.actor, p.question, p.trainee_id)
        if result.outcome is CoachOutcome.REFERRAL:
            return ToolOutput(
                "An exceptional incident: the procedure is shown instead.",
                "referral",
                stop=REFERRAL,
                cost_usd=result.cost_usd,
            )
        if result.outcome is CoachOutcome.NOT_SENT or result.advice is None:
            return ToolOutput(
                "The coach did not answer: the question was about health, which Pacer does not advise on.",
                "not sent (health)",
                cost_usd=result.cost_usd,
            )
        advice = result.advice
        by_passage = {c.passage: c for c in advice.citations}
        lines = ["Understanding:", advice.understanding, "Steps:"]
        for number, step in enumerate(advice.steps, start=1):
            basis = (
                "source: " + "; ".join(by_passage[pid].source for pid in step.passages if pid in by_passage)
                if step.grounded
                else "general advice, no source in the library"
            )
            lines.append(f"{number}. {step.action} ({basis})")
            if step.example:
                lines.append(f"   For example: {step.example}")
        if advice.questions:
            lines += ["Worth asking:", *(f"- {q}" for q in advice.questions)]
        if advice.involve:
            lines += [
                "Involve:",
                *(f"- {role.label}: {why}" if why else f"- {role.label}" for role, why in advice.involve),
            ]
        sources = [SourceRef(c.source_id, c.source, c.evidence, c.section) for c in advice.citations if c.cited]
        grounded = sum(step.grounded for step in advice.steps)
        return ToolOutput(
            "\n".join(lines),
            f"{len(advice.steps)} steps, {grounded} grounded",
            sources=sources,
            cost_usd=result.cost_usd,
        )

    def report_incident(self, ctx: ToolContext, p: ReportIncident) -> ToolOutput:
        return ToolOutput("The procedure is shown.", "referral", stop=REFERRAL)


# --- helpers -------------------------------------------------------------------------------------


def _person(trainee: Trainee) -> Person:
    return Person(PersonKind.TRAINEE, trainee.key, trainee.first_name, trainee.last_name)


def _matches(trainee: Trainee, wanted: str) -> bool:
    """A full name, a first name or a last name, as written (the model passes a token, restored here)."""
    words = " ".join(wanted.split())
    return words in (trainee.full_name, trainee.first_name, trainee.last_name)


def _current_cohorts(cohorts: list[Cohort], day: date) -> list[Cohort]:
    """The cohorts running on this day; if none is, the latest one (the demo's year may have ended)."""
    running = [c for c in cohorts if c.start_date <= day <= c.end_date]
    if running:
        return running
    return sorted(cohorts, key=lambda c: c.start_date)[-1:]


def _attendance_lines(s: AttendanceSummary) -> list[str]:
    """The rate on one line, and each session missed most on its own line: a title about health is then
    removed alone, not with the trainee's rate (the coach does the same, review 2026-10-26)."""
    if s.rate is None:
        return [f"- {s.trainee.full_name} (trainee_id {s.trainee.key}): nothing recorded yet"]
    lines = [
        f"- {s.trainee.full_name} (trainee_id {s.trainee.key}): {round(s.rate * 100)}% "
        f"(present {s.present} of {s.counted}; excused {s.excused}; unmarked {s.unmarked})"
    ]
    # Template sessions only: a one-off's title is free text written for one day, kept off any model.
    lines += [f"  - missed: {title} ({n})" for title, n in list(s.missed_in_template.items())[:3]]
    return lines
