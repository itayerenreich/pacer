"""The counselor coach: practical educational advice about a trainee, grounded where it can be.

    coach.ask(actor, question, trainee_id=None)  ->  CoachResult (a referral, advice, or not sent)

    0. permissions  may this actor see the trainee? Otherwise 404, before anything else happens.
    1. safety       SafetyGate on the QUESTION (phrases, then a model). An exceptional incident gets the
                    fixed procedure and nothing else (decision 2026-10-25: the gate checks the question
                    only; an approved note like "separated after a fight" must not turn every question
                    about that trainee into a referral).
    2. context      what the academy knows about the trainee, built in code (services/coach_context.py)
    3. minimize     health sentences out, then the question and the context anonymized as ONE text, so
                    every token means the same person everywhere in it
    4. retrieve     library passages for the anonymized question, on this computer (bge-m3)
    5. model        Claude, coach_advice@v1, structured: understanding, questions, steps, who to involve
    6. validate     in code: cited passages must have been sent; a step's grounding is DECIDED by its valid
                    citations (none = "general advice", decision 2026-10-25: grounded where it can be,
                    marked where it cannot); nothing about health; tokens must be known
    7. answer       names restored, and the procedure line added to every answer, in code

Unlike /library/ask, no passage is not "no answer": the coach still advises, and every step says
whether it rests on a source. The library is small; honesty about grounding is what makes that safe.

No transaction is open while a model works: one short read, then models.
"""

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from pacer.domain import StaffMember
from pacer.repositories import ScoredChunk, UnitOfWork
from pacer.services import CoachContextService, TraineeContext

from .anonymization import (
    AnonymizationError,
    AnonymizedText,
    Anonymizer,
    PersonKind,
    Roster,
    deanonymize,
    load_roster,
    tokens_in,
)
from .answer import Citation, passage_ids
from .embeddings import Embedder
from .issues import Issue, IssueCode
from .library import retrieve
from .prompts import Prompt, load_prompt
from .risk_screen import SafetyGate
from .routing import AITask, Router
from .safety import Referral
from .sensitive import is_sensitive, remove_sensitive_sentences
from .structured import complete_structured

log = logging.getLogger(__name__)

PROMPT_VERSION = 1  # pinned: a used version is never edited, a change is a new version

# Added by code to every answer: the gate can miss, and a sentence about health never reaches it.
PROCEDURE_NOTE = (
    "אם עולה חשש לאירוע חריג (פגיעה עצמית, אלימות, פגיעה בחניך), התשובה הזו לא מתאימה: "
    "פועלים לפי נוהל המכינה ומעדכנים מיד את המנהל."
)
# When nothing of the question is left after the health sentences are removed.
NOT_SENT_NOTE = (
    "השאלה עוסקת במידע בריאותי, ש-Pacer לא שולח למודל ולא מייעץ בו. "
    "כדאי להתייעץ עם המנהל או עם גורם מקצועי."
)
WITHHELD_UNDERSTANDING = "(הסבר המצב הושמט: עסק בבריאות)"


# --- what the model must return ------------------------------------------------------------------


class InvolveRole(StrEnum):
    MANAGER = "manager"
    PARENTS = "parents"
    ACADEMY_COUNSELOR = "academy_counselor"
    PROFESSIONAL = "professional"

    @property
    def label(self) -> str:
        return _ROLE_LABELS[self]


_ROLE_LABELS = {
    InvolveRole.MANAGER: "מנהל/ת המכינה",
    InvolveRole.PARENTS: "ההורים",
    InvolveRole.ACADEMY_COUNSELOR: "יועץ/ת המכינה",
    InvolveRole.PROFESSIONAL: "גורם מקצועי",
}


class CoachStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: str = Field(min_length=5, max_length=600, description="One clear action, in Hebrew")
    example: str | None = Field(
        default=None, max_length=300, description="One sentence the counselor could say to the trainee, or null"
    )
    passages: list[str] = Field(
        max_length=4, description="Ids of the passages this step rests on (P1...); empty for general practice"
    )


class Involve(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: InvolveRole
    why: str = Field(min_length=3, max_length=300, description="One Hebrew sentence")


class CoachAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    understanding: str = Field(min_length=10, max_length=1500, description="Hebrew, 2 to 4 sentences")
    questions: list[str] = Field(max_length=3, description="Short clarifying questions, only when it matters")
    steps: list[CoachStep] = Field(min_length=1, max_length=6, description="2 to 5 practical steps, in order")
    involve: list[Involve] = Field(max_length=4, description="Who else should be part of this; often empty")


# --- the message: built with names, then minimized and anonymized as one text --------------------


def coach_document(question: str, context: TraineeContext | None) -> str:
    """The question and what is known about the trainee, WITH names. One line per item, so that a
    line about health is removed on its own (remove_sensitive_sentences splits at line breaks)."""
    lines = ["Question:", question, ""]
    if context is None:
        return "\n".join([*lines, "Trainee: none (a general question)"])
    lines += [f"Trainee: {context.trainee.full_name}", "Goals:"]
    for item in context.goals:
        goal = item.goal
        lines.append(f"- {goal.title} (category: {goal.category}, progress: {goal.progress}%, status: {goal.status})")
        lines += [f"  - {u.created_at:%Y-%m-%d}: {u.progress}% · {u.note}" for u in item.recent_updates]
    if not context.goals:
        lines.append("- (no goals yet)")
    lines.append("Notes approved by counselors (oldest first):")
    lines += [f"- [{note.kind}] {note.created_at:%Y-%m-%d}: {note.text}" for note in context.notes]
    if not context.notes:
        lines.append("- (no notes yet)")
    return "\n".join(lines)


@dataclass(frozen=True)
class PreparedMessage:
    document: AnonymizedText  # the question and the trainee, anonymized: what the model reads
    question: AnonymizedText  # the question alone, anonymized: what retrieval searches with
    removed: int  # health sentences removed before anything was sent
    question_left: bool  # False when the whole question was about health: nothing is sent


def prepare_coach_message(
    roster: Roster, question: str, context: TraineeContext | None
) -> PreparedMessage:
    """The exact text that leaves the computer. One function for the coach AND the dry run."""
    anonymizer = Anonymizer(roster)
    prefer = []
    if context is not None and (person := roster.find(PersonKind.TRAINEE, context.trainee.key)):
        prefer.append(person)
    kept_question, _ = remove_sensitive_sentences(question)
    kept_document, removed = remove_sensitive_sentences(coach_document(question, context))
    return PreparedMessage(
        document=anonymizer.anonymize(kept_document, prefer=prefer),
        question=anonymizer.anonymize(kept_question, prefer=prefer),
        removed=removed,
        question_left=bool(kept_question.strip()),
    )


def coach_message(document: AnonymizedText, hits: Sequence[ScoredChunk]) -> str:
    """The anonymized document (the type proves it), then the library passages, each with its id."""
    if not isinstance(document, AnonymizedText):
        raise TypeError("coach_message() needs AnonymizedText, from Anonymizer.anonymize()")
    lines = [document.text, "", "Passages:" if hits else "Passages: none (no source in the library is close)"]
    for pid, hit in zip(passage_ids(list(hits)), hits, strict=True):
        where = " · ".join(p for p in (hit.chunk.section, hit.chunk.pages) if p)
        lines += [
            "",
            f"[{pid}] {hit.source.citation} · {hit.source.evidence_level.label}" + (f" · {where}" if where else ""),
            hit.chunk.text,
        ]
    return "\n".join(lines)


# --- content validation --------------------------------------------------------------------------


@dataclass(frozen=True)
class AdviceStep:
    action: str
    example: str | None
    passages: list[str]  # valid ids only; empty = general practice

    @property
    def grounded(self) -> bool:
        return bool(self.passages)


@dataclass(frozen=True)
class Advice:
    understanding: str
    questions: list[str]
    steps: list[AdviceStep]
    involve: list[tuple[InvolveRole, str]]
    citations: list[Citation]
    issues: list[Issue]


_PASSAGE_ID = re.compile(r"^P\d{1,2}$")


def validate_coach(raw: CoachAnswer, hits: Sequence[ScoredChunk], document: AnonymizedText) -> Advice:
    sent = set(passage_ids(list(hits)))
    known_tokens = set(document.people) | set(document.unresolved)
    issues: list[Issue] = []

    def restore(text: str, item: int | None, section: str | None) -> str:
        if invented := sorted(set(tokens_in(text)) - known_tokens):
            issues.append(Issue(IssueCode.UNKNOWN_TOKEN, item, f"left as written: {', '.join(invented)}", section))
        return deanonymize(text, document)

    def health(item: int | None, section: str) -> None:
        issues.append(Issue(IssueCode.SENSITIVE_TEXT, item, "health or diagnosis; not shown", section))

    if is_sensitive(raw.understanding):
        health(None, "understanding")
        understanding = WITHHELD_UNDERSTANDING
    else:
        understanding = restore(raw.understanding, None, "understanding")

    questions = []
    for number, question in enumerate(raw.questions, start=1):
        if is_sensitive(question):
            health(number, "questions")
        else:
            questions.append(restore(question, number, "questions"))

    steps: list[AdviceStep] = []
    for number, step in enumerate(raw.steps, start=1):
        if is_sensitive(step.action) or (step.example is not None and is_sensitive(step.example)):
            health(number, "steps")
            continue
        ids = [p.strip() for p in step.passages]
        if unknown := [p for p in ids if not _PASSAGE_ID.match(p) or p not in sent]:
            detail = f"cited {', '.join(unknown)}, never sent; removed. The step counts as general advice"
            issues.append(Issue(IssueCode.UNKNOWN_PASSAGE, number, detail, "steps"))
        steps.append(
            AdviceStep(
                action=restore(step.action, number, "steps"),
                example=None if step.example is None else restore(step.example, number, "steps"),
                passages=list(dict.fromkeys(p for p in ids if p in sent)),
            )
        )
    if not steps:
        issues.append(Issue(IssueCode.NO_ADVICE, None, "no step left to show", "steps"))

    involve: list[tuple[InvolveRole, str]] = []
    for number, item in enumerate(raw.involve, start=1):
        if any(role is item.role for role, _ in involve):
            issues.append(Issue(IssueCode.DUPLICATE, number, f"{item.role} twice; kept the first", "involve"))
        elif is_sensitive(item.why):
            health(number, "involve")
            involve.append((item.role, ""))  # the role stays: who to involve matters more than why
        else:
            involve.append((item.role, restore(item.why, number, "involve")))

    cited = {p for step in steps for p in step.passages}
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
        for pid, hit in zip(passage_ids(list(hits)), hits, strict=True)
    ]
    return Advice(understanding, questions, steps, involve, citations, issues)


# --- the pipeline --------------------------------------------------------------------------------


class CoachOutcome(StrEnum):
    REFERRAL = "referral"  # an exceptional incident: the procedure, no advice
    ADVICE = "advice"
    NOT_SENT = "not_sent"  # the whole question was about health: nothing left the computer


@dataclass
class CoachResult:
    outcome: CoachOutcome
    question: str
    trainee_id: int | None
    referral: Referral | None = None
    advice: Advice | None = None
    note: str = PROCEDURE_NOTE
    prompt_ref: str | None = None  # None when no advice model was asked
    cost_usd: float | None = 0.0  # the safety screen and the advice together
    removed_sentences: int = 0
    passages: list[str] = field(default_factory=list, repr=False)  # the text of P1, P2... (the evaluation)


def _total(*costs: float | None) -> float | None:
    return None if any(c is None for c in costs) else sum(c for c in costs if c is not None)


class Coach:
    """uow: short transactions (permissions, context, roster, search). None is open while a model works."""

    def __init__(
        self,
        uow: UnitOfWork,
        router: Router,
        embedder: Embedder,
        chunking: str,
        *,
        gate: SafetyGate | None = None,
        k: int = 4,
        prompt: Prompt | None = None,
    ) -> None:
        self._uow = uow
        self._router = router
        self._embedder = embedder
        self._chunking = chunking
        self._gate = gate or SafetyGate(router)
        self._k = k
        self._prompt = prompt or load_prompt("coach_advice", PROMPT_VERSION)

    def ask(self, actor: StaffMember, question: str, trainee_id: int | None = None) -> CoachResult:
        if not question.strip():
            raise ValueError("The question is empty")

        # 0 + 2. Permissions and context, in one short transaction (404 before anything else).
        with self._uow() as repos:
            context = None
            if trainee_id is not None:
                context = CoachContextService(
                    repos.trainees, repos.groups, repos.goals, repos.goal_updates, repos.notes
                ).build(actor, trainee_id)
            roster = load_roster(repos.staff, repos.trainees)

        # 1. Safety, on the question.
        about = roster.find(PersonKind.TRAINEE, trainee_id) if trainee_id is not None else None
        decision = self._gate.assess(question, Anonymizer(roster), prefer=(about,) if about else ())
        if decision.is_risk:
            log.warning("Coach: referral (by %s) for trainee %s, staff %s", decision.found_by, trainee_id, actor.id)
            return CoachResult(
                CoachOutcome.REFERRAL, question, trainee_id, referral=decision.referral, cost_usd=decision.cost_usd
            )

        # 3. Minimize and anonymize.
        prepared = prepare_coach_message(roster, question, context)
        if not prepared.question_left:
            return CoachResult(
                CoachOutcome.NOT_SENT,
                question,
                trainee_id,
                note=NOT_SENT_NOTE,
                cost_usd=decision.cost_usd,
                removed_sentences=prepared.removed,
            )
        if Anonymizer(roster).contains_known_name(prepared.document.text):  # defense in depth
            raise AnonymizationError("A known name is in the coach's message; nothing was sent")

        # 4. Retrieve, 5. the model, 6. validate.
        hits = retrieve(
            self._uow, self._embedder, self._chunking, prepared.question.text, k=self._k, for_cloud_model=True
        )
        client = self._router.client_for(AITask.COACH_ADVICE)
        max_tokens = self._router.route(AITask.COACH_ADVICE).max_tokens
        result = complete_structured(
            client, self._prompt, coach_message(prepared.document, hits), CoachAnswer, max_tokens=max_tokens
        )
        advice = validate_coach(result.value, hits, prepared.document)
        log.info(
            "Coach: advice for trainee %s, asked by staff %s: steps=%s grounded=%s involve=%s issues=%s removed=%s",
            trainee_id,
            actor.id,
            len(advice.steps),
            sum(step.grounded for step in advice.steps),
            ",".join(role for role, _ in advice.involve) or "-",
            len(advice.issues),
            prepared.removed,
        )
        return CoachResult(
            CoachOutcome.ADVICE,
            question,
            trainee_id,
            advice=advice,
            prompt_ref=result.prompt_ref,
            cost_usd=_total(decision.cost_usd, result.cost_usd),
            removed_sentences=prepared.removed,
            passages=[hit.chunk.text for hit in hits],
        )
