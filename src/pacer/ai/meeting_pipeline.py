"""The staff meeting pipeline: from a recording or a transcript to proposals the manager reviews.

    UPLOADED ──transcribe──> TRANSCRIBED ──anonymize──> ANONYMIZED ──extract──> PENDING_APPROVAL
    (audio)    local Whisper  (raw text)    code, local  (tokens)    Claude +    (proposals)
                                                                     validation

Every stage reads its input from the database and saves its output there, and moves the meeting
one status forward. So the status always says what has been done, and when a stage fails (the
model is down, its answer was invalid twice) the meeting stays where it was: running the pipeline
again continues from that stage, without redoing (or paying for) the stages before it.

Three rules shape this module:

1. No model call inside a database transaction. A model takes seconds (Whisper, minutes). A
   transaction held open that long would keep a connection and locks busy for everyone else.
   So a stage that calls a model does: short transaction (read) -> model -> short transaction
   (write). The write locks the meeting row and checks the status again: if another request
   finished the same stage meanwhile, this answer is discarded instead of saved twice.
2. Only anonymized text reaches the cloud. extraction_message() accepts an AnonymizedText,
   the type only the Anonymizer can create, and checks the final message for known names once
   more before it leaves this computer.
3. The model proposes, code checks, a person decides. The answer is validated for SHAPE
   (complete_structured, Pydantic) and for CONTENT here (validate_extraction: is the owner a
   participant? is the date sensible? a duplicate?). What content validation cannot fix is
   cleared and reported as an issue, never guessed. Then the manager approves each proposal.
"""

import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from pacer.domain import MeetingStatus, ProposedTask, StaffMeeting, StaffMember, StaffRole
from pacer.domain.meeting import DESCRIPTION_MAX_LENGTH, TRANSCRIPT_MIN_LENGTH
from pacer.repositories import Repos, UnitOfWork
from pacer.services import MeetingService, RuleViolationError, TaskDraft

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
from .prompts import Prompt, load_prompt
from .routing import AITask, PrivacyViolation, Router
from .structured import complete_structured
from .transcription import RecordingStore, Transcriber

log = logging.getLogger(__name__)

M = MeetingStatus
_STAGES = list(MeetingStatus)
DUE_DATE_HORIZON = timedelta(days=366)  # a deadline more than a year after the meeting is a misread date
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


# --- what the model must return (validated for shape by complete_structured) ----------------------


class ExtractedTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(
        min_length=5,
        max_length=250,  # names are longer than tokens: room for them, under the service's 300
        description="One Hebrew sentence starting with an infinitive verb, understandable on its own",
    )
    owner: str | None = Field(
        description="Token of the participant who took or was given the task, e.g. [STAFF_3]; null if unclear"
    )
    due_date: date | None = Field(description="Deadline as YYYY-MM-DD, only if one was said; otherwise null")


class MeetingExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        min_length=20, max_length=3000, description="Hebrew, 3 to 8 sentences: topics and decisions"
    )
    tasks: list[ExtractedTask] = Field(max_length=30, description="Each task once; empty if there were none")


# --- the message: anonymized text only ---------------------------------------------------------


_ROLE = {StaffRole.MANAGER: "manager", StaffRole.COUNSELOR: "counselor"}


def extraction_message(transcript: AnonymizedText, held_on: date, participants: Sequence[StaffMember]) -> str:
    """The user message for the model. Takes AnonymizedText, not str: raw text cannot get here.

    The header is in English on purpose: it is fixed text, and Hebrew words in it ("שני" in
    "יום שני") could collide with a name in the roster and trip the final name check.
    """
    if not isinstance(transcript, AnonymizedText):
        raise TypeError("extraction_message() needs AnonymizedText, from Anonymizer.anonymize()")
    lines = [
        f"Meeting date: {held_on.isoformat()} ({_WEEKDAYS[held_on.weekday()]})",  # not strftime: locale-free
        "Participants:",
        *(f"- [{PersonKind.STAFF}_{s.id}]: {_ROLE[s.role]}" for s in participants),
        "",
        "Transcript:",
        transcript.text,
    ]
    return "\n".join(lines)


# --- content validation: the model cannot know the world, the database can ------------------------


class IssueCode(StrEnum):
    UNKNOWN_TOKEN = "unknown_token"  # the model wrote a token that was not in what it received
    OWNER_INVALID = "owner_invalid"  # not a token at all
    OWNER_AMBIGUOUS = "owner_ambiguous"  # [NAME_n]: the name fits more than one person
    OWNER_NOT_STAFF = "owner_not_staff"  # a trainee cannot own a staff task
    OWNER_NOT_PARTICIPANT = "owner_not_participant"
    DUE_BEFORE_MEETING = "due_before_meeting"
    DUE_TOO_FAR = "due_too_far"
    DUPLICATE = "duplicate"
    DESCRIPTION_SHORTENED = "description_shortened"


@dataclass(frozen=True)
class Issue:
    """Something validation changed in the model's answer. Tokens and numbers only, never names:
    issues go to logs and to the API."""

    code: IssueCode
    task: int | None  # 1-based position in the model's list; None for the summary
    detail: str


@dataclass(frozen=True)
class ValidatedExtraction:
    summary: str  # names restored
    drafts: list[TaskDraft]
    issues: list[Issue]


_STAFF_TOKEN = re.compile(r"^\[STAFF_(\d+)\]$")
_TRAINEE_TOKEN = re.compile(r"^\[TRAINEE_\d+\]$")
_NAME_TOKEN = re.compile(r"^\[NAME_\d+\]$")


def validate_extraction(
    extraction: MeetingExtraction, source: AnonymizedText, *, held_on: date, participant_ids: Iterable[int]
) -> ValidatedExtraction:
    """Check the answer against the world, and turn tokens back into names (on this computer)."""
    participants = set(participant_ids)
    known_tokens = set(source.people) | set(source.unresolved)
    issues: list[Issue] = []

    def unknown_tokens(text: str, task: int | None) -> None:
        if invented := sorted(set(tokens_in(text)) - known_tokens):
            issues.append(Issue(IssueCode.UNKNOWN_TOKEN, task, f"left as written: {', '.join(invented)}"))

    unknown_tokens(extraction.summary, None)
    summary = deanonymize(extraction.summary, source)

    drafts: list[TaskDraft] = []
    seen: set[tuple[str, int | None]] = set()
    for number, task in enumerate(extraction.tasks, start=1):
        owner_id = _owner_id(task.owner, participants, number, issues)
        due_date = _due_date(task.due_date, held_on, number, issues)

        unknown_tokens(task.description, number)
        description = " ".join(deanonymize(task.description, source).split())
        if len(description) > DESCRIPTION_MAX_LENGTH:
            description = description[: DESCRIPTION_MAX_LENGTH - 1].rstrip() + "…"
            issues.append(Issue(IssueCode.DESCRIPTION_SHORTENED, number, f"cut to {DESCRIPTION_MAX_LENGTH} characters"))

        key = (description.casefold(), owner_id)
        if key in seen:
            issues.append(Issue(IssueCode.DUPLICATE, number, "same description and owner as an earlier task; dropped"))
            continue
        seen.add(key)
        drafts.append(TaskDraft(description=description, owner_id=owner_id, due_date=due_date))

    return ValidatedExtraction(summary=summary, drafts=drafts, issues=issues)


def _owner_id(owner: str | None, participants: set[int], task: int, issues: list[Issue]) -> int | None:
    if owner is None:
        return None
    owner = owner.strip()
    if match := _STAFF_TOKEN.match(owner):
        staff_id = int(match.group(1))
        if staff_id in participants:
            return staff_id
        issues.append(Issue(IssueCode.OWNER_NOT_PARTICIPANT, task, f"{owner} did not attend; owner cleared"))
    elif _TRAINEE_TOKEN.match(owner):
        issues.append(Issue(IssueCode.OWNER_NOT_STAFF, task, f"{owner} is a trainee; owner cleared"))
    elif _NAME_TOKEN.match(owner):
        issues.append(Issue(IssueCode.OWNER_AMBIGUOUS, task, f"{owner} fits more than one person; owner cleared"))
    else:
        issues.append(Issue(IssueCode.OWNER_INVALID, task, "the owner was not a participant token; owner cleared"))
    return None


def _due_date(due: date | None, held_on: date, task: int, issues: list[Issue]) -> date | None:
    if due is None:
        return None
    if due < held_on:
        issues.append(Issue(IssueCode.DUE_BEFORE_MEETING, task, f"{due} is before the meeting; date cleared"))
        return None
    if due > held_on + DUE_DATE_HORIZON:
        issues.append(Issue(IssueCode.DUE_TOO_FAR, task, f"{due} is over a year away; date cleared"))
        return None
    return due


# --- the pipeline ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtractionResult:
    meeting: StaffMeeting
    proposals: list[ProposedTask]
    issues: list[Issue]
    prompt_ref: str
    attempts: int
    cost_usd: float | None


class MeetingPipeline:
    """Runs the stages. Only a manager may run them (the same rule as reading a meeting).

    uow:          a factory of short transactions (repositories(engine) in production, fakes in tests)
    router:       hands out the model for AITask.EXTRACT_MEETING, and enforces the privacy rule
    transcriber:  local speech-to-text, needed only for meetings uploaded as audio
    recordings:   where uploaded audio waits for transcription
    """

    def __init__(
        self,
        uow: UnitOfWork,
        router: Router,
        *,
        transcriber: Transcriber | None = None,
        recordings: RecordingStore | None = None,
        prompt: Prompt | None = None,
    ) -> None:
        self._uow = uow
        self._router = router
        self._transcriber = transcriber
        self._recordings = recordings
        self._prompt = prompt or load_prompt("meeting_extract", 1)  # a pinned version, not "latest"

    def process(self, actor: StaffMember, meeting_id: int) -> ExtractionResult:
        """Run every stage that has not run yet, continuing from where the meeting is."""
        status = self._read(actor, meeting_id).status
        if _STAGES.index(status) >= _STAGES.index(M.EXTRACTED):
            raise RuleViolationError(f"Meeting {meeting_id} is {status}: there is nothing left to process")
        if status is M.UPLOADED:
            self.transcribe(actor, meeting_id)
        if status in (M.UPLOADED, M.TRANSCRIBED):
            self.anonymize(actor, meeting_id)
        return self.extract(actor, meeting_id)

    # --- recordings --------------------------------------------------------------------------

    def attach_recording(
        self, actor: StaffMember, meeting_id: int, chunks: Iterable[bytes], suffix: str
    ) -> int:
        """Store the audio of a meeting that is waiting for it (created without a transcript).
        Returns its size in bytes. Refused up front when this server cannot transcribe."""
        self._require_local_transcriber()
        _require_status(self._read(actor, meeting_id), M.UPLOADED)
        return self._require_recordings().save(meeting_id, chunks, suffix).stat().st_size

    # --- stage 1: transcribe (local) ---------------------------------------------------------

    def transcribe(self, actor: StaffMember, meeting_id: int) -> StaffMeeting:
        transcriber = self._require_local_transcriber()
        recordings = self._require_recordings()
        _require_status(self._read(actor, meeting_id), M.UPLOADED)
        recording = recordings.find(meeting_id)
        if recording is None:
            raise RuleViolationError(f"Meeting {meeting_id} has no recording yet: upload one first")

        text = transcriber.transcribe(recording).strip()  # no transaction open: this takes minutes
        if len(text) < TRANSCRIPT_MIN_LENGTH:
            raise RuleViolationError(
                f"The recording of meeting {meeting_id} gave almost no text (silence?). It was kept; "
                "check it and upload it again"
            )
        with self._uow() as repos:
            meeting = _require_status(repos.meetings.lock(meeting_id), M.UPLOADED)
            transcribed = repos.meetings.update(meeting.advance(M.TRANSCRIBED, transcript=text))
        # Only after the transcript is committed: if saving had failed, the recording is still here.
        recordings.delete(meeting_id)
        log.info("Meeting %s transcribed: %s characters", meeting_id, len(text))
        return transcribed

    # --- stage 2: anonymize (local, code only) -------------------------------------------------

    def anonymize(self, actor: StaffMember, meeting_id: int) -> StaffMeeting:
        with self._uow() as repos:
            self._service(repos).get_meeting(actor, meeting_id)  # permission
            meeting = _require_status(repos.meetings.lock(meeting_id), M.TRANSCRIBED)
            safe = Anonymizer(load_roster(repos.staff, repos.trainees)).anonymize(meeting.transcript)
            anonymized = repos.meetings.update(meeting.advance(M.ANONYMIZED, anonymized_transcript=safe.text))
        if self._recordings is not None:
            self._recordings.delete(meeting_id)  # a recording never outlives its transcript, even after a crash
        log.info(
            "Meeting %s anonymized: %s people, %s unresolved names",
            meeting_id,
            len(safe.people),
            len(safe.unresolved),
        )
        return anonymized

    # --- stage 3: extract (cloud, anonymized only) and validate ----------------------------------

    def extract(self, actor: StaffMember, meeting_id: int) -> ExtractionResult:
        client = self._router.client_for(AITask.EXTRACT_MEETING)  # fails now if, e.g., there is no key
        max_tokens = self._router.route(AITask.EXTRACT_MEETING).max_tokens

        # 1. Read, in a short transaction.
        with self._uow() as repos:
            meeting = _require_status(self._service(repos).get_meeting(actor, meeting_id), M.ANONYMIZED)
            roster = load_roster(repos.staff, repos.trainees)
            participants = [s for i in meeting.participant_ids if (s := repos.staff.get(i)) is not None]

        # 2. Anonymize again, from the transcript. The token -> person mapping is needed to read the
        #    answer, and it is never stored. Same roster -> same text as the ANONYMIZED stage; if a
        #    person was added since, the new text is safer, and it is the one saved below.
        safe = anonymize_meeting(roster, meeting.transcript, participants)
        message = extraction_message(safe, meeting.held_on, participants)
        if Anonymizer(roster).contains_known_name(message):  # defense in depth, right before sending
            raise AnonymizationError(f"Meeting {meeting_id}: a known name is in the message; nothing was sent")

        # 3. The model, with NO transaction open. Then validation, which needs no database either.
        result = complete_structured(client, self._prompt, message, MeetingExtraction, max_tokens=max_tokens)
        validated = validate_extraction(
            result.value, safe, held_on=meeting.held_on, participant_ids=meeting.participant_ids
        )

        # 4. Write, in a short transaction, with the meeting locked: the summary and the proposals
        #    are saved together, or not at all.
        with self._uow() as repos:
            current = repos.meetings.lock(meeting_id)
            if current is None or current.status is not M.ANONYMIZED:
                raise RuleViolationError(
                    f"Meeting {meeting_id} was processed by another request while the model was working; "
                    "this answer was discarded"
                )
            repos.meetings.update(
                current.advance(M.EXTRACTED, anonymized_transcript=safe.text, summary=validated.summary)
            )
            proposals = self._service(repos).submit_proposals(meeting_id, validated.drafts)
            final = repos.meetings.get(meeting_id)

        log.info(
            "Meeting %s extracted: prompt=%s attempts=%s proposals=%s issues=%s cost=%s",
            meeting_id,
            result.prompt_ref,
            result.attempts,
            len(proposals),
            len(validated.issues),
            "unknown" if result.cost_usd is None else f"${result.cost_usd:.6f}",
        )
        return ExtractionResult(
            meeting=final,
            proposals=proposals,
            issues=validated.issues,
            prompt_ref=result.prompt_ref,
            attempts=result.attempts,
            cost_usd=result.cost_usd,
        )

    # --- helpers -------------------------------------------------------------------------------

    def _read(self, actor: StaffMember, meeting_id: int) -> StaffMeeting:
        with self._uow() as repos:
            return self._service(repos).get_meeting(actor, meeting_id)

    @staticmethod
    def _service(repos: Repos) -> MeetingService:
        return MeetingService(repos.meetings, repos.proposals, repos.tasks, repos.staff)

    def _require_local_transcriber(self) -> Transcriber:
        if self._transcriber is None:
            raise RuleViolationError("Audio transcription is not set up on this server; upload a transcript")
        if not self._transcriber.runs_locally:  # the same rule as the router: personal data stays here
            raise PrivacyViolation("Recordings may only be transcribed by a model on this computer")
        return self._transcriber

    def _require_recordings(self) -> RecordingStore:
        if self._recordings is None:
            raise RuleViolationError("Audio recordings are not set up on this server; upload a transcript")
        return self._recordings


def _require_status(meeting: StaffMeeting | None, status: MeetingStatus) -> StaffMeeting:
    if meeting is None:
        raise RuleViolationError("The meeting no longer exists")
    if meeting.status is not status:
        raise RuleViolationError(f"Meeting {meeting.id} is {meeting.status}; this stage needs {status}")
    return meeting


def anonymize_meeting(roster: Roster, transcript: str, participants: Sequence[StaffMember]) -> AnonymizedText:
    """The transcript with names replaced, and the participants known by token even if never named."""
    known = [roster.find(PersonKind.STAFF, s.id) for s in participants]
    return Anonymizer(roster).anonymize(transcript, known=[p for p in known if p is not None])
