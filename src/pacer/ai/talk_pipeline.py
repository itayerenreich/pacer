"""The personal talk pipeline: a counselor's notes become a summary, goal progress proposals, and
notes for the trainee's long-term memory. The counselor decides on each one.

    RECORDED ──summarize──> SUMMARIZED ──(the counselor decides on every proposal)──> REVIEWED
              anonymize, Claude, validate

The same three rules as the meeting pipeline (see meeting_pipeline.py):
1. No model call inside a database transaction: short read -> model -> short write, and the write
   locks the talk and checks its status again.
2. Only anonymized text reaches the cloud, and not all of it. First, every sentence of the notes
   that touches health or a diagnosis is REMOVED, here, on this computer (decision 2026-10-18:
   data minimization; the counselor's own notes keep it, in the local database). Then the whole
   message, header, goal titles and notes, is built and anonymized as ONE text, so a name in a
   goal title is caught too, and every [NAME_n] means the same word everywhere in it. The talk's
   trainee and counselor are "preferred": an ambiguous first name in their own talk is them.
3. The model proposes, code checks, a person decides. Content validation here: the goal must be
   one of this trainee's goals and still open, the progress must change something, no duplicates,
   and nothing about health or a diagnosis is ever proposed to be kept, neither as a note nor as
   a progress reason (which becomes a permanent GoalUpdate note): a word list, as a second line
   of defense behind the prompt.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from pacer.domain import (
    Goal,
    GoalStatus,
    NoteKind,
    PersonalTalk,
    StaffMember,
    TalkProposal,
    TalkProposalKind,
    TalkStatus,
    Trainee,
)
from pacer.domain.talk import NOTE_MAX_LENGTH, REASON_MAX_LENGTH
from pacer.repositories import Repos, UnitOfWork
from pacer.services import GoalService, RuleViolationError, TalkDraft, TalkService
from pacer.services.errors import found

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
from .issues import Issue, IssueCode
from .prompts import Prompt, load_prompt
from .routing import AITask, Router
from .sensitive import is_sensitive, remove_sensitive_sentences
from .structured import complete_structured

log = logging.getLogger(__name__)

_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


# --- what the model must return ---------------------------------------------------------------------


class ProgressSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_id: int = Field(description="The id of one of the goals in the message")
    progress: int = Field(ge=0, le=100, description="The new progress, 0 to 100")
    reason: str = Field(min_length=5, max_length=400, description="One Hebrew sentence: the evidence")


class NoteSuggestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: NoteKind
    text: str = Field(min_length=5, max_length=250, description="One Hebrew sentence")


class TalkExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=20, max_length=2000, description="Hebrew, 3 to 6 sentences")
    progress: list[ProgressSuggestion] = Field(max_length=10, description="Only goals the talk touched")
    notes: list[NoteSuggestion] = Field(max_length=5, description="Educational notes worth remembering")


# --- the message: built with names, then anonymized as one text --------------------------------------


def talk_document(
    trainee: Trainee, counselor: StaffMember, held_on: date, goals: Sequence[Goal], notes: str
) -> str:
    """The full message WITH names. It never leaves this function's caller un-anonymized."""
    goal_lines = [
        f"- goal_id {g.id}: {_goal_title(g)} (category: {g.category}, progress: {g.progress}%, status: {g.status})"
        for g in goals
    ] or ["(no goals yet)"]
    return "\n".join(
        [
            f"Talk date: {held_on.isoformat()} ({_WEEKDAYS[held_on.weekday()]})",
            f"Trainee: {trainee.full_name}",
            f"Counselor: {counselor.full_name}",
            "Goals:",
            *goal_lines,
            "",
            "The counselor's notes:",
            notes,
        ]
    )


WITHHELD_TITLE = "(title withheld)"


def _goal_title(goal: Goal) -> str:
    """A goal title about health is not sent either ("להגיע לכל פגישה עם הפסיכולוגית")."""
    return WITHHELD_TITLE if is_sensitive(goal.title) else goal.title


def anonymize_talk(
    roster: Roster,
    trainee: Trainee,
    counselor: StaffMember,
    held_on: date,
    goals: Sequence[Goal],
    notes: str,
) -> AnonymizedText:
    """The whole message, anonymized as one text. The talk's trainee and counselor are preferred
    when a name is ambiguous: in their own talk, it is most likely them."""
    about = [roster.find(PersonKind.TRAINEE, trainee.key), roster.find(PersonKind.STAFF, counselor.key)]
    return Anonymizer(roster).anonymize(
        talk_document(trainee, counselor, held_on, goals, notes), prefer=[p for p in about if p is not None]
    )


def prepare_talk_message(
    roster: Roster,
    trainee: Trainee,
    counselor: StaffMember,
    held_on: date,
    goals: Sequence[Goal],
    notes: str,
) -> tuple[AnonymizedText, int]:
    """The exact message that goes to the model, and how many sensitive sentences (and goal titles)
    were removed.
    One function for the pipeline AND the dry run, so the dry run never shows something else."""
    kept, removed = remove_sensitive_sentences(notes)
    removed += sum(1 for goal in goals if is_sensitive(goal.title))
    return anonymize_talk(roster, trainee, counselor, held_on, goals, kept), removed


# --- content validation -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidatedTalk:
    summary: str  # names restored
    drafts: list[TalkDraft]
    issues: list[Issue]


def validate_talk(extraction: TalkExtraction, source: AnonymizedText, *, goals: Sequence[Goal]) -> ValidatedTalk:
    known_tokens = set(source.people) | set(source.unresolved)
    by_id = {goal.id: goal for goal in goals}
    issues: list[Issue] = []

    def restore(text: str, item: int | None, section: str | None, limit: int) -> str:
        if invented := sorted(set(tokens_in(text)) - known_tokens):
            issues.append(Issue(IssueCode.UNKNOWN_TOKEN, item, f"left as written: {', '.join(invented)}", section))
        restored = " ".join(deanonymize(text, source).split())
        if len(restored) > limit:
            issues.append(Issue(IssueCode.DESCRIPTION_SHORTENED, item, f"cut to {limit} characters", section))
            restored = restored[: limit - 1].rstrip() + "…"
        return restored

    summary = restore(extraction.summary, None, None, limit=10_000)
    drafts: list[TalkDraft] = []

    seen_goals: set[int] = set()
    for number, suggestion in enumerate(extraction.progress, start=1):
        goal = by_id.get(suggestion.goal_id)
        if goal is None:
            issues.append(Issue(IssueCode.GOAL_NOT_FOUND, number, f"goal {suggestion.goal_id}; dropped", "progress"))
            continue
        if goal.status is GoalStatus.DROPPED:
            issues.append(Issue(IssueCode.GOAL_CLOSED, number, f"goal {goal.id} was dropped; dropped", "progress"))
            continue
        if suggestion.progress == goal.progress:
            detail = f"goal {goal.id} is already at {goal.progress}%"
            issues.append(Issue(IssueCode.NO_CHANGE, number, detail, "progress"))
            continue
        if goal.id in seen_goals:
            issues.append(Issue(IssueCode.DUPLICATE, number, f"goal {goal.id} twice; kept the first", "progress"))
            continue
        if is_sensitive(suggestion.reason):  # the reason would stay forever in the goal's history
            detail = f"goal {goal.id}: health or diagnosis; not kept"
            issues.append(Issue(IssueCode.SENSITIVE_TEXT, number, detail, "progress"))
            continue
        seen_goals.add(goal.key)
        reason = restore(suggestion.reason, number, "progress", REASON_MAX_LENGTH)
        drafts.append(
            TalkDraft(kind=TalkProposalKind.PROGRESS, goal_id=goal.id, progress=suggestion.progress, text=reason)
        )

    seen_notes: set[str] = set()
    for number, note in enumerate(extraction.notes, start=1):
        if is_sensitive(note.text):  # the detail says nothing about what the note said
            issues.append(Issue(IssueCode.SENSITIVE_TEXT, number, "health or diagnosis; not kept", "notes"))
            continue
        text = restore(note.text, number, "notes", NOTE_MAX_LENGTH)
        if text.casefold() in seen_notes:
            issues.append(Issue(IssueCode.DUPLICATE, number, "same note as an earlier one; dropped", "notes"))
            continue
        seen_notes.add(text.casefold())
        drafts.append(TalkDraft(kind=TalkProposalKind.NOTE, note_kind=note.kind, text=text))

    return ValidatedTalk(summary=summary, drafts=drafts, issues=issues)


# --- the pipeline ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TalkResult:
    talk: PersonalTalk
    proposals: list[TalkProposal]
    issues: list[Issue]
    prompt_ref: str
    attempts: int
    cost_usd: float | None
    removed_sentences: int = 0  # about health, removed before sending; the counselor's notes keep them


class TalkPipeline:
    """Runs the summary. Only someone who may edit the trainee's goals may run it."""

    def __init__(self, uow: UnitOfWork, router: Router, *, prompt: Prompt | None = None) -> None:
        self._uow = uow
        self._router = router
        self._prompt = prompt or load_prompt("talk_summary", 2)

    def summarize(self, actor: StaffMember, talk_id: int) -> TalkResult:
        client = self._router.client_for(AITask.SUMMARIZE_TALK)
        max_tokens = self._router.route(AITask.SUMMARIZE_TALK).max_tokens

        # 1. Read, in a short transaction.
        with self._uow() as repos:
            talk, trainee = talk_service(repos).talk_to_summarize(actor, talk_id)
            _require_status(talk, TalkStatus.RECORDED)
            counselor = found(repos.staff.get(talk.counselor_id), "Staff member", talk.counselor_id)
            goals = repos.goals.list_by_trainee(trainee.key)
            roster = load_roster(repos.staff, repos.trainees)

        # 2. Sensitive sentences out, then one document, anonymized as a whole. Only safe.text goes
        #    to the model.
        safe, removed = prepare_talk_message(roster, trainee, counselor, talk.held_on, goals, talk.notes)
        if Anonymizer(roster).contains_known_name(safe.text):  # defense in depth, right before sending
            raise AnonymizationError(f"Talk {talk_id}: a known name is in the message; nothing was sent")

        # 3. The model, with NO transaction open. Then validation.
        result = complete_structured(client, self._prompt, safe.text, TalkExtraction, max_tokens=max_tokens)
        validated = validate_talk(result.value, safe, goals=goals)

        # 4. Write, in a short transaction, with the talk locked.
        with self._uow() as repos:
            current = repos.talks.lock(talk_id)
            if current is None or current.status is not TalkStatus.RECORDED:
                raise RuleViolationError(
                    f"Talk {talk_id} was summarized by another request meanwhile; this answer was discarded"
                )
            repos.talks.update(
                current.advance(TalkStatus.SUMMARIZED, anonymized_notes=safe.text, summary=validated.summary)
            )
            proposals = talk_service(repos).submit_proposals(talk_id, validated.drafts)
            final = found(repos.talks.get(talk_id), "Talk", talk_id)

        log.info(
            "Talk %s summarized: prompt=%s attempts=%s proposals=%s issues=%s removed_sentences=%s cost=%s",
            talk_id,
            result.prompt_ref,
            result.attempts,
            len(proposals),
            len(validated.issues),
            removed,
            "unknown" if result.cost_usd is None else f"${result.cost_usd:.6f}",
        )
        return TalkResult(
            talk=final,
            proposals=proposals,
            issues=validated.issues,
            prompt_ref=result.prompt_ref,
            attempts=result.attempts,
            cost_usd=result.cost_usd,
            removed_sentences=removed,
        )


def talk_service(repos: Repos) -> TalkService:
    goals = GoalService(repos.goals, repos.goal_updates, repos.trainees, repos.groups)
    return TalkService(
        repos.talks, repos.talk_proposals, repos.notes, repos.trainees, repos.groups, repos.goals, goals
    )


def _require_status(talk: PersonalTalk, status: TalkStatus) -> None:
    if talk.status is not status:
        raise RuleViolationError(f"Talk {talk.id} is {talk.status}; it can be summarized only when {status}")
