"""Staff meetings, the tasks an AI proposes from them, and the tasks a manager approves.

Three state machines, one idea: the AI proposes, a person decides.

    StaffMeeting:   transcribed -> anonymized -> extracted -> pending_approval -> approved
    ProposedTask:   proposed -> approved | rejected          (a manager decides; may edit first)
    Task:           open -> done

A ProposedTask is never shown to counselors. Only an approved one becomes a Task, with the
manager recorded as the one who created it.
"""

from datetime import UTC, date, datetime
from typing import Self

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from .entity import Entity
from .enums import MeetingStatus, ProposalStatus, TaskStatus
from .errors import DomainRuleError
from .state import require_transition

M, P, T = MeetingStatus, ProposalStatus, TaskStatus

MEETING_TRANSITIONS: dict[MeetingStatus, frozenset[MeetingStatus]] = {
    M.TRANSCRIBED: frozenset({M.ANONYMIZED}),
    M.ANONYMIZED: frozenset({M.EXTRACTED}),
    M.EXTRACTED: frozenset({M.PENDING_APPROVAL}),
    M.PENDING_APPROVAL: frozenset({M.APPROVED}),
    M.APPROVED: frozenset(),
}
PROPOSAL_TRANSITIONS: dict[ProposalStatus, frozenset[ProposalStatus]] = {
    P.PROPOSED: frozenset({P.APPROVED, P.REJECTED}),
    P.APPROVED: frozenset(),
    P.REJECTED: frozenset(),
}
TASK_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    T.OPEN: frozenset({T.DONE}),
    T.DONE: frozenset(),
}

_ORDER = list(MeetingStatus)
DESCRIPTION_MAX_LENGTH = 300  # a task's description, from the model's proposal to the real Task
TRANSCRIPT_MIN_LENGTH = 20


def _now() -> datetime:
    return datetime.now(UTC)


class StaffMeeting(Entity):
    model_config = ConfigDict(validate_assignment=True)

    title: str = Field(min_length=2, max_length=200)
    held_on: date
    created_by_id: int
    participant_ids: list[int] = Field(min_length=1)
    status: MeetingStatus = MeetingStatus.TRANSCRIBED
    # Raw text: PERSONAL data. Stays in the local database; never sent to a cloud model.
    transcript: str = Field(min_length=TRANSCRIPT_MIN_LENGTH, repr=False)
    # Names replaced by tokens: the only form a cloud model may see.
    anonymized_transcript: str | None = Field(default=None, repr=False)
    summary: str | None = Field(default=None, repr=False)
    created_at: AwareDatetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def check_fields_match_status(self) -> Self:
        if len(set(self.participant_ids)) != len(self.participant_ids):
            raise ValueError("A participant appears more than once")
        reached = _ORDER.index(self.status)
        for needed, field_name in [
            (M.ANONYMIZED, "anonymized_transcript"),
            (M.EXTRACTED, "summary"),
        ]:
            if reached >= _ORDER.index(needed) and getattr(self, field_name) is None:
                raise ValueError(f"A meeting that is {self.status} must have a {field_name}")
        return self

    def advance(self, status: MeetingStatus, **fields) -> Self:
        """A NEW meeting in the next status, with the fields that status needs."""
        require_transition(MEETING_TRANSITIONS, self.status, status, "meeting")
        return self.model_validate({**self.model_dump(), **fields, "status": status})


class ProposedTask(Entity):
    """A task the AI proposed. Waits for the manager, who may edit it before approving."""

    model_config = ConfigDict(validate_assignment=True)

    meeting_id: int
    description: str = Field(min_length=5, max_length=DESCRIPTION_MAX_LENGTH)
    owner_id: int | None = None  # None when the model's answer named no one we could resolve
    due_date: date | None = None
    status: ProposalStatus = ProposalStatus.PROPOSED
    decided_by_id: int | None = None
    decided_at: AwareDatetime | None = None
    task_id: int | None = None  # the Task it became, once approved
    created_at: AwareDatetime = Field(default_factory=_now)

    @model_validator(mode="after")
    def check_decision_fields(self) -> Self:
        decided = self.status is not P.PROPOSED
        if decided != (self.decided_by_id is not None and self.decided_at is not None):
            raise ValueError("A decided proposal says who decided and when; an open one does not")
        if (self.status is P.APPROVED) != (self.task_id is not None):
            raise ValueError("Only an approved proposal points to a task, and it must")
        if self.status is P.APPROVED and self.owner_id is None:
            raise ValueError("An approved proposal must have an owner")
        return self

    def edited(self, **changes) -> Self:
        """A NEW proposal with the manager's edits (description, owner_id, due_date)."""
        if self.status is not P.PROPOSED:
            raise DomainRuleError(f"A proposal that is {self.status} can no longer be edited")
        allowed = {"description", "owner_id", "due_date"}
        if unknown := set(changes) - allowed:
            raise ValueError(f"Cannot edit {sorted(unknown)}")
        return self.model_validate({**self.model_dump(), **changes})

    def approve(self, manager_id: int, task_id: int) -> Self:
        return self._decide(P.APPROVED, manager_id, task_id=task_id)

    def reject(self, manager_id: int) -> Self:
        return self._decide(P.REJECTED, manager_id)

    def _decide(self, status: ProposalStatus, manager_id: int, **fields) -> Self:
        require_transition(PROPOSAL_TRANSITIONS, self.status, status, "proposal")
        return self.model_validate(
            {
                **self.model_dump(),
                **fields,
                "status": status,
                "decided_by_id": manager_id,
                "decided_at": _now(),
            }
        )


class Task(Entity):
    """A real task, owned by one staff member. It comes from a staff meeting (a proposal the manager
    approved), or is written by hand (2026-10-27): by a staff member for themselves, or by the manager
    for anyone."""

    model_config = ConfigDict(validate_assignment=True)

    description: str = Field(min_length=5, max_length=DESCRIPTION_MAX_LENGTH)
    owner_id: int
    due_date: date | None = None
    status: TaskStatus = TaskStatus.OPEN
    created_by_id: int  # the manager who approved it, or whoever wrote it by hand
    source_meeting_id: int | None = None  # None: written by hand
    trainee_id: int | None = None  # the trainee it is about, if any: shown in their profile too
    created_at: AwareDatetime = Field(default_factory=_now)
    completed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def check_completion(self) -> Self:
        if (self.status is T.DONE) != (self.completed_at is not None):
            raise ValueError("A done task has a completion time; an open one does not")
        return self

    def complete(self) -> Self:
        require_transition(TASK_TRANSITIONS, self.status, T.DONE, "task")
        return self.model_validate({**self.model_dump(), "status": T.DONE, "completed_at": _now()})

    def with_due_date(self, due_date: date | None) -> Self:
        """A new deadline (or none). Only while the task is open: a done task's history stays as it was."""
        if self.status is not T.OPEN:
            raise DomainRuleError(f"Task {self.id} is {self.status}: its deadline no longer changes")
        return self.model_validate({**self.model_dump(), "due_date": due_date})

    @property
    def written_by_hand(self) -> bool:
        return self.source_meeting_id is None
