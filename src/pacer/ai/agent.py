"""The agent: a model that decides which tools to call, in a loop that our code runs (stage 6).

Until stage 6 every AI feature in Pacer was a fixed pipeline: the code decides the steps. Here the
model decides them, and the code decides everything else:

    agent.chat(actor, message, history)  ->  AgentResult (an answer, a referral, not sent, or stopped)

    0. phrases      the safety phrase list on every user turn, locally, before anything else
    1. one read     the roster (every name to hide) and which trainees this actor may see
    2. screen       the model screen of SafetyGate on the new message: an exceptional incident gets the
                    fixed procedure, and the loop never starts
    3. minimize     health sentences out of every turn, then anonymized in ONE token space for the whole
                    conversation (AnonymizationSession). Nothing left of the message: nothing is sent
    4. the loop     the model answers or asks for tools; our code runs each tool with the actor bound,
                    removes health sentences from the result, anonymizes it, and sends it back. Limits:
                    rounds, tool calls, and cost. The last round must answer in text
    5. the answer   health sentences out, unknown tokens reported, names restored (a trainee the asker
                    may not see only as the asker wrote them), the procedure line added by code

The history comes from the client (decision 2026-10-27: no conversation is stored). It is untrusted
text like the message itself: checked by the phrase list, minimized and anonymized again every time.
Only the user and assistant texts are kept; past tool calls are not, so the model fetches again what
it needs.

No transaction is open while a model works: every tool opens its own short one.
"""

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pacer.domain import StaffMember, StaffRole, Weekday, academy_today
from pacer.repositories import UnitOfWork
from pacer.services.permissions import can_view_trainee

from .anonymization import (
    AnonymizationError,
    AnonymizationSession,
    Anonymizer,
    Person,
    deanonymize,
    load_roster,
    tokens_in,
)
from .coach import PROCEDURE_NOTE, names_as_written
from .issues import Issue, IssueCode
from .llm_client import ChatClient, ChatResponse
from .prompts import Prompt, load_prompt
from .risk_screen import SafetyGate
from .routing import AITask, Router
from .safety import Referral, check_safety, referral_for
from .sensitive import HEALTH_MASK, is_sensitive, remove_sensitive_sentences
from .tools import SourceRef, ToolContext, ToolRegistry, ToolRun

log = logging.getLogger(__name__)

PROMPT_VERSION = 1

MAX_ROUNDS = 6  # model calls per message; the last one must answer in text
MAX_TOOL_CALLS = 10
MAX_COST_USD = 0.25  # per message, models inside tools (the coach) included
HISTORY_TURNS = 10  # the client may send more; only the last ones are read

NOT_SENT_NOTE = (
    "ההודעה עוסקת במידע בריאותי, ש-Pacer לא שולח למודל ולא מייעץ בו. "
    "כדאי להתייעץ עם המנהל או עם גורם מקצועי. " + PROCEDURE_NOTE
)
STOPPED_ANSWER = "לא הצלחתי להשלים את התשובה הפעם. אפשר לנסות לשאול שוב, בשאלה ממוקדת יותר."
EMPTY_LEFT = "(nothing is left of this after removing health information)"


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str


class AgentOutcome(StrEnum):
    ANSWER = "answer"
    REFERRAL = "referral"  # an exceptional incident (phrases, the screen, the agent, or the coach)
    NOT_SENT = "not_sent"  # the whole message was about health: nothing left the computer
    STOPPED = "stopped"  # a limit was reached before an answer


@dataclass(frozen=True)
class AgentStep:
    """One tool call, as the counselor sees it under the answer: what was looked up, and how it went."""

    tool: str
    arguments: dict[str, Any]  # names restored the way the asker may see them
    ok: bool
    summary: str  # counts and ids, never content
    latency_ms: int


@dataclass
class AgentResult:
    outcome: AgentOutcome
    answer: str | None = None
    referral: Referral | None = None
    note: str = PROCEDURE_NOTE
    steps: list[AgentStep] = field(default_factory=list)
    sources: list[SourceRef] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    stopped_by: str | None = None  # which limit (rounds, tools, cost) or "empty", when outcome is STOPPED
    model: str | None = None
    prompt_ref: str | None = None
    rounds: int = 0
    cost_usd: float | None = 0.0
    removed_sentences: int = 0


# An unknown cost (a model missing from the price table) is reported as unknown, but the CAP cannot
# ignore it: that would turn the cap off exactly when nobody knows what is spent (review, 2026-10-27).
# For the cap, each unknown part counts as this much: more than any priced call of ours.
UNKNOWN_COST_FOR_CAP_USD = 0.05


class _Cost:
    """A running total that becomes None (unknown) as soon as one part is unknown, never a silent 0."""

    def __init__(self) -> None:
        self.known = 0.0
        self.unknown = 0

    def add(self, cost: float | None) -> None:
        if cost is None:
            self.unknown += 1
        else:
            self.known += cost

    @property
    def total(self) -> float | None:
        return None if self.unknown else self.known

    @property
    def for_cap(self) -> float:
        return self.known + self.unknown * UNKNOWN_COST_FOR_CAP_USD


class Agent:
    def __init__(
        self,
        uow: UnitOfWork,
        router: Router,
        tools: ToolRegistry,
        *,
        gate: SafetyGate | None = None,
        prompt: Prompt | None = None,
        today: Callable[[], date] = academy_today,
        max_rounds: int = MAX_ROUNDS,
        max_tool_calls: int = MAX_TOOL_CALLS,
        max_cost_usd: float = MAX_COST_USD,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self._uow = uow
        self._router = router
        self._tools = tools
        self._gate = gate or SafetyGate(router)
        self._prompt = prompt or load_prompt("agent", PROMPT_VERSION)
        self._today = today
        self._max_rounds = max_rounds
        self._max_tool_calls = max_tool_calls
        self._max_cost = max_cost_usd

    def chat(self, actor: StaffMember, message: str, history: Sequence[Turn] = ()) -> AgentResult:
        if not message.strip():
            raise ValueError("The message is empty")
        history = list(history)[-HISTORY_TURNS:]
        while history and history[0].role == "assistant":  # a conversation starts with the user
            history.pop(0)

        # 0. The phrase list, on what the counselor wrote now and before. Local, and it reveals nothing.
        found = check_safety(message, *(t.content for t in history if t.role == "user"))
        if found.is_risk:
            log.warning("Agent: referral (by phrases), staff %s", actor.id)
            return AgentResult(AgentOutcome.REFERRAL, referral=referral_for(found))

        # 1. One short read: who exists (names to hide) and whom this actor may see.
        with self._uow() as repos:
            roster = load_roster(repos.staff, repos.trainees)
            visible = {
                t.key
                for t in repos.trainees.list_all()
                if (group := repos.groups.get(t.group_id)) is not None and can_view_trainee(actor, group)
            }

        # 2. The model screen, on the new message.
        cost = _Cost()
        decision = self._gate.assess(message, Anonymizer(roster))
        cost.add(decision.cost_usd)
        if decision.is_risk:
            log.warning("Agent: referral (by %s), staff %s", decision.found_by, actor.id)
            return AgentResult(AgentOutcome.REFERRAL, referral=decision.referral, cost_usd=cost.total)

        # 3. Minimize and anonymize every turn, in one token space. What the staff member wrote is marked
        #    as theirs (asker=True): a trainee they may not see is written back only in their own words.
        session = AnonymizationSession(roster)
        kept, removed = remove_sensitive_sentences(message)
        if not kept.strip():
            return AgentResult(
                AgentOutcome.NOT_SENT, note=NOT_SENT_NOTE, cost_usd=cost.total, removed_sentences=removed
            )
        messages: list[dict[str, Any]] = []
        for turn in history:
            text, n = remove_sensitive_sentences(turn.content)
            removed += n
            content = session.anonymize(text, asker=turn.role == "user") if text.strip() else EMPTY_LEFT
            messages.append({"role": turn.role, "content": content})
        question = session.anonymize(kept, asker=True)
        messages.append({"role": "user", "content": self._context_line(actor, session) + question})
        name_for = names_as_written(session.written, visible)

        # 4. The loop.
        client = self._client()
        max_tokens = self._router.route(AITask.AGENT).max_tokens
        context = ToolContext(actor=actor, session=session, today=self._today(), name_for=name_for)
        result = AgentResult(
            AgentOutcome.STOPPED, model=client.model, prompt_ref=self._prompt.ref, removed_sentences=removed
        )
        runs: list[ToolRun] = []
        forced: str | None = None  # a limit that was reached: the next round is the last, in text only
        try:
            for round_number in range(1, self._max_rounds + 1):
                result.rounds = round_number
                if forced is None and cost.for_cap >= self._max_cost:
                    forced = "cost"
                last = round_number == self._max_rounds or forced is not None
                response = client.chat(
                    self._prompt, messages, tools=self._tools.schemas(), max_tokens=max_tokens, allow_tools=not last
                )
                cost.add(response.cost_usd)
                if not response.tool_calls:
                    self._finish(result, response, session, name_for)
                    if forced is not None and result.outcome is AgentOutcome.ANSWER:
                        detail = f"the {forced} limit was reached: answered from what was found until then"
                        result.issues.append(Issue(IssueCode.LIMIT_REACHED, None, detail, "answer"))
                    log.info(
                        "Agent: %s for staff %s: rounds=%s tools=%s issues=%s removed=%s",
                        result.outcome,
                        actor.id,
                        result.rounds,
                        ",".join(run.name for run in runs) or "-",
                        len(result.issues),
                        result.removed_sentences,
                    )
                    return result
                if last:  # asked to answer in text, it asked for a tool anyway
                    result.stopped_by = forced or "rounds"
                    break
                messages.append(_assistant_message(response))
                for call in response.tool_calls:
                    # The limits are checked before EVERY call, not only between rounds: one round may ask
                    # for ten coach calls, each with models of its own (review, 2026-10-27).
                    if forced is None and len(runs) >= self._max_tool_calls:
                        forced = "tools"
                    if forced is None and cost.for_cap >= self._max_cost:
                        forced = "cost"
                    if forced is not None:
                        # Every call must get a result, or the provider rejects the conversation.
                        messages.append(_tool_message(call.id, f"Error: the {forced} limit was reached. Answer now."))
                        continue
                    run = self._tools.run(context, call.name, call.arguments)
                    runs.append(run)
                    cost.add(run.output.cost_usd)
                    if run.output.stop is not None:
                        log.warning("Agent: referral (by the %s tool), staff %s", run.name, actor.id)
                        result.outcome, result.referral = AgentOutcome.REFERRAL, run.output.stop
                        return result
                    messages.append(_tool_message(call.id, self._tool_text(run, session, result)))
            log.warning("Agent: stopped by the %s limit, staff %s", result.stopped_by, actor.id)
            result.answer = STOPPED_ANSWER
            return result
        finally:
            result.cost_usd = cost.total
            result.steps = [_step(run, session, name_for) for run in runs]
            result.sources = _sources(runs)

    def _tool_text(self, run: ToolRun, session: AnonymizationSession, result: AgentResult) -> str:
        """A tool's result as the model may read it: health sentences out, then anonymized. If the
        anonymizer cannot vouch for it, the model is told so and reads nothing of it."""
        text, n = remove_sensitive_sentences(run.output.text)
        result.removed_sentences += n
        if not text.strip():
            return EMPTY_LEFT
        try:
            return session.anonymize(text, prefer=run.output.about)
        except AnonymizationError:
            log.error("Agent: the result of %s could not be anonymized; it was not sent", run.name)
            return "Error: this result could not be checked for names, so it was not sent. Answer without it."

    def _client(self) -> ChatClient:
        client = self._router.client_for(AITask.AGENT)  # the router's privacy checks, as for every task
        if not isinstance(client, ChatClient):
            raise TypeError(f"The agent's model {client.model} cannot be given tools")
        return client

    def _context_line(self, actor: StaffMember, session: AnonymizationSession) -> str:
        """Today's date and who is asking, as the model needs them (the actor's name as a token)."""
        today = self._today()
        role = "the manager" if actor.role is StaffRole.MANAGER else "a counselor"
        asker = session.anonymize(actor.full_name)
        return f"(Today is {Weekday(today.weekday()).name.title()} {today}. Asked by {asker}, {role}.)\n\n"

    def _finish(
        self,
        result: AgentResult,
        response: ChatResponse,
        session: AnonymizationSession,
        name_for: Callable[[Person], str],
    ) -> None:
        text, removed = remove_sensitive_sentences(response.text.strip())
        if removed:
            result.issues.append(
                Issue(IssueCode.SENSITIVE_TEXT, None, f"{removed} sentence(s) about health; not shown", "answer")
            )
        source = session.view()
        known = set(source.people) | set(source.unresolved)
        if invented := sorted(set(tokens_in(text)) - known):
            result.issues.append(
                Issue(IssueCode.UNKNOWN_TOKEN, None, f"left as written: {', '.join(invented)}", "answer")
            )
        if response.truncated:
            result.issues.append(
                Issue(IssueCode.TRUNCATED, None, "the answer was cut off at the token limit", "answer")
            )
        if not text.strip():
            result.stopped_by = "empty"
            result.answer = STOPPED_ANSWER
            return
        result.outcome = AgentOutcome.ANSWER
        result.answer = deanonymize(text, source, name_for)


def _assistant_message(response: ChatResponse) -> dict[str, Any]:
    """The model's turn, sent back as it was, except arguments that are not a JSON object: LiteLLM's
    translation for Anthropic fails on them, and the tool's error result already says what was wrong."""
    return {
        "role": "assistant",
        "content": response.text or None,
        "tool_calls": [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": _json_object_or_empty(c.arguments)},
            }
            for c in response.tool_calls
        ],
    }


def _json_object_or_empty(arguments: str) -> str:
    try:
        return arguments if isinstance(json.loads(arguments or "{}"), dict) else "{}"
    except json.JSONDecodeError:
        return "{}"


def _tool_message(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def _step(run: ToolRun, session: AnonymizationSession, name_for: Callable[[Person], str]) -> AgentStep:
    """The arguments as the asker may see them: tokens back to names with the same rule as the answer,
    and a text about health (the model wrote it) not shown, as in the answer (review, 2026-10-27)."""
    source = session.view()

    def restore(value: Any) -> Any:
        if isinstance(value, str):
            text = deanonymize(value, source, name_for)
            return HEALTH_MASK if is_sensitive(text) else text
        if isinstance(value, list):
            return [restore(v) for v in value]
        if isinstance(value, dict):
            return {k: restore(v) for k, v in value.items()}
        return value

    return AgentStep(run.name, restore(run.arguments), run.ok, run.output.summary, run.latency_ms)


def _sources(runs: Sequence[ToolRun]) -> list[SourceRef]:
    """Every library source a tool showed the model, once each, in the order they came."""
    seen: dict[tuple[int, str | None], SourceRef] = {}
    for run in runs:
        for ref in run.output.sources:
            seen.setdefault((ref.source_id, ref.section), ref)
    return list(seen.values())
