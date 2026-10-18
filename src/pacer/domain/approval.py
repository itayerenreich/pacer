"""Approval requests: a counselor asks the manager before a sensitive action.

ApprovalRequest is a small state machine:

    PENDING ──approve──> APPROVED
       │
       └────reject────> REJECTED

APPROVED and REJECTED are final. Every allowed move is listed in TRANSITIONS, and
any other move raises InvalidTransitionError. approve() and reject() return a NEW
request; the original object never changes.
"""

from datetime import UTC, datetime
from typing import Self

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from .entity import Entity
from .enums import ApprovalKind, ApprovalStatus, TraineeStatus
from .errors import InvalidTransitionError

TRANSITIONS: dict[ApprovalStatus, frozenset[ApprovalStatus]] = {
    ApprovalStatus.PENDING: frozenset({ApprovalStatus.APPROVED, ApprovalStatus.REJECTED}),
    ApprovalStatus.APPROVED: frozenset(),
    ApprovalStatus.REJECTED: frozenset(),
}


class ApprovalRequest(Entity):
    model_config = ConfigDict(validate_assignment=True)

    kind: ApprovalKind
    trainee_id: int
    requested_by_id: int
    reason: str = Field(min_length=3, max_length=1000)
    # Only for STATUS_CHANGE: the status the counselor wants the trainee moved to.
    requested_status: TraineeStatus | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    decided_by_id: int | None = None
    decision_note: str | None = Field(default=None, max_length=1000)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    decided_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def check_requested_status(self) -> Self:
        is_status_change = self.kind == ApprovalKind.STATUS_CHANGE
        if is_status_change and self.requested_status is None:
            raise ValueError("A status change request must say which status it asks for")
        if not is_status_change and self.requested_status is not None:
            raise ValueError("Only a status change request has a requested_status")
        return self

    @model_validator(mode="after")
    def check_decision_fields(self) -> Self:
        decided = self.decided_by_id is not None and self.decided_at is not None
        if self.status == ApprovalStatus.PENDING and (self.decided_by_id or self.decided_at):
            raise ValueError("A pending request has no decision yet")
        if self.status != ApprovalStatus.PENDING and not decided:
            raise ValueError("A decided request must say who decided and when")
        return self

    @property
    def is_pending(self) -> bool:
        return self.status == ApprovalStatus.PENDING

    def approve(self, manager_id: int, note: str | None = None) -> Self:
        return self._decide(ApprovalStatus.APPROVED, manager_id, note)

    def reject(self, manager_id: int, note: str | None = None) -> Self:
        return self._decide(ApprovalStatus.REJECTED, manager_id, note)

    def _decide(self, new_status: ApprovalStatus, manager_id: int, note: str | None) -> Self:
        if not TRANSITIONS[self.status]:
            raise InvalidTransitionError(f"This request is already {self.status}")
        if new_status not in TRANSITIONS[self.status]:
            raise InvalidTransitionError(
                f"Cannot move an approval request from {self.status} to {new_status}"
            )
        return self.model_validate(
            {
                **self.model_dump(),
                "status": new_status,
                "decided_by_id": manager_id,
                "decision_note": note,
                "decided_at": datetime.now(UTC),
            }
        )
