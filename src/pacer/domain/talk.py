"""Personal talks between a counselor and a trainee, what the AI proposes from them, and the
approved notes that become the trainee's long-term memory.

The same idea as staff meetings: the AI proposes, a person decides.

    PersonalTalk:   recorded -> summarized -> reviewed
    TalkProposal:   proposed -> approved | rejected       (the counselor may edit first)
        kind=progress  approved -> a regular GoalUpdate, through GoalService
        kind=note      approved -> a TraineeNote

TraineeNote is deliberately small and curated (memory option B): only what a counselor approved,
in short educational observations, never diagnoses or medical information. Later conversations
load these notes, not whole transcripts.

Data minimization: a decided proposal keeps no text. Approved content lives in what it became
(the GoalUpdate or the TraineeNote, so deleting the note really deletes it); a rejected guess of
the model about a person is not kept anywhere.
"""

from datetime import UTC, date, datetime
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from .enums import NoteKind, ProposalStatus, TalkProposalKind, TalkStatus
from .errors import DomainRuleError
from .state import require_transition

TS, P, K = TalkStatus, ProposalStatus, TalkProposalKind

TALK_TRANSITIONS: dict[TalkStatus, frozenset[TalkStatus]] = {
    TS.RECORDED: frozenset({TS.SUMMARIZED}),
    TS.SUMMARIZED: frozenset({TS.REVIEWED}),
    TS.REVIEWED: frozenset(),
}
TALK_PROPOSAL_TRANSITIONS: dict[ProposalStatus, frozenset[ProposalStatus]] = {
    P.PROPOSED: frozenset({P.APPROVED, P.REJECTED}),
    P.APPROVED: frozenset(),
    P.REJECTED: frozenset(),
}
NOTE_MAX_LENGTH = 300
REASON_MAX_LENGTH = 500


def _now() -> datetime:
    return datetime.now(UTC)


class PersonalTalk(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: int | None = None
    trainee_id: int
    counselor_id: int  # who held the talk and wrote the notes
    held_on: date
    status: TalkStatus = TalkStatus.RECORDED
    # The counselor's notes: PERSONAL data. Stays in the local database; never sent as is.
    notes: str = Field(min_length=20, max_length=20_000, repr=False)
    anonymized_notes: str | None = Field(default=None, repr=False)  # exactly what the model saw
    summary: str | None = Field(default=None, repr=False)
    created_at: AwareDatetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def check_fields_match_status(self) -> Self:
        summarized = self.status is not TS.RECORDED
        if summarized and (self.summary is None or self.anonymized_notes is None):
            raise ValueError(f"A talk that is {self.status} must have a summary and the anonymized notes")
        return self

    def advance(self, status: TalkStatus, **fields) -> Self:
        require_transition(TALK_TRANSITIONS, self.status, status, "talk")
        return self.model_validate({**self.model_dump(), **fields, "status": status})


class TalkProposal(BaseModel):
    """One proposal from a talk: a goal progress update, or a note. Waits for the counselor."""

    model_config = ConfigDict(validate_assignment=True)

    id: int | None = None
    talk_id: int
    kind: TalkProposalKind
    # kind=progress
    goal_id: int | None = None
    progress: int | None = Field(default=None, ge=0, le=100)
    # kind=note
    note_kind: NoteKind | None = None
    # progress: why (becomes the GoalUpdate's note). note: the note itself. None once decided.
    text: str | None = Field(default=None, min_length=5, max_length=REASON_MAX_LENGTH, repr=False)
    status: ProposalStatus = ProposalStatus.PROPOSED
    decided_by_id: int | None = None
    decided_at: AwareDatetime | None = None
    goal_update_id: int | None = None  # the GoalUpdate it became (progress, approved)
    note_id: int | None = None  # the TraineeNote it became (note, approved)
    created_at: AwareDatetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def check_fields_match_kind_and_status(self) -> Self:
        if self.kind is K.PROGRESS:
            if self.goal_id is None or self.progress is None or self.note_kind is not None:
                raise ValueError("A progress proposal has a goal and a progress value, and no note kind")
        elif self.note_kind is None or self.goal_id is not None or self.progress is not None:
            raise ValueError("A note proposal has a note kind, and no goal or progress")
        if self.kind is K.NOTE and self.text is not None and len(self.text) > NOTE_MAX_LENGTH:
            raise ValueError(f"A note is at most {NOTE_MAX_LENGTH} characters")
        decided = self.status is not P.PROPOSED
        if decided != (self.decided_by_id is not None and self.decided_at is not None):
            raise ValueError("A decided proposal says who decided and when; an open one does not")
        if decided == (self.text is not None):
            raise ValueError("An open proposal has its text; a decided one keeps none")
        if self.status is not P.APPROVED and (self.goal_update_id is not None or self.note_id is not None):
            raise ValueError("Only an approved proposal points to what it became")
        if self.status is P.APPROVED and self.kind is K.PROGRESS and self.goal_update_id is None:
            raise ValueError("An approved progress proposal points to its GoalUpdate")
        if (self.kind is K.NOTE and self.goal_update_id is not None) or (
            self.kind is K.PROGRESS and self.note_id is not None
        ):
            raise ValueError("A proposal becomes what its kind says")
        # An approved NOTE may have note_id None: the counselor deleted that note later.
        return self

    def edited(self, **changes) -> Self:
        """A NEW proposal with the counselor's edits: text, and progress or note_kind (by kind)."""
        if self.status is not P.PROPOSED:
            raise DomainRuleError(f"A proposal that is {self.status} can no longer be edited")
        allowed = {"text", "progress"} if self.kind is K.PROGRESS else {"text", "note_kind"}
        if unknown := set(changes) - allowed:  # a rule of this object, not a malformed value: 409
            raise DomainRuleError(f"Cannot edit {sorted(unknown)} of a {self.kind} proposal")
        return self.model_validate({**self.model_dump(), **changes})

    def approve(self, counselor_id: int, *, goal_update_id: int | None = None, note_id: int | None = None) -> Self:
        return self._decide(P.APPROVED, counselor_id, goal_update_id=goal_update_id, note_id=note_id)

    def reject(self, counselor_id: int) -> Self:
        return self._decide(P.REJECTED, counselor_id)

    def _decide(self, status: ProposalStatus, counselor_id: int, **fields) -> Self:
        require_transition(TALK_PROPOSAL_TRANSITIONS, self.status, status, "proposal")
        return self.model_validate(
            {
                **self.model_dump(),
                **fields,
                "status": status,
                "decided_by_id": counselor_id,
                "decided_at": _now(),
                "text": None,  # the content now lives in what it became, or nowhere
            }
        )


class TraineeNote(BaseModel):
    """Approved long-term memory about a trainee: short, educational, attributed."""

    model_config = ConfigDict(validate_assignment=True)

    id: int | None = None
    trainee_id: int
    kind: NoteKind
    text: str = Field(min_length=5, max_length=NOTE_MAX_LENGTH)
    author_id: int  # the counselor who wrote or approved it
    source_talk_id: int | None = None  # None when written by hand
    created_at: AwareDatetime = Field(default_factory=_now)
