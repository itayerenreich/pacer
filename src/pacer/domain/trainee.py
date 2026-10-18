"""Trainees and their sensitive personal files."""

from datetime import UTC, datetime
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from .enums import TraineeStatus
from .errors import DomainRuleError


class Trainee(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: int | None = None
    first_name: str = Field(min_length=1, max_length=50)
    last_name: str = Field(min_length=1, max_length=50)
    cohort_id: int
    group_id: int
    primary_counselor_id: int
    status: TraineeStatus = TraineeStatus.ACTIVE
    # When the file was closed (LEFT or GRADUATED). Starts the retention clock for the trainee's
    # notes: after the period the academy sets, they are deleted (services/retention.py).
    closed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _closed_at_matches_status(self) -> Self:
        if self.is_closed and self.closed_at is None:
            raise DomainRuleError(f"A {self.status} trainee needs closed_at: when the file was closed")
        if not self.is_closed and self.closed_at is not None:
            raise DomainRuleError(f"A {self.status} trainee has an open file: closed_at must be empty")
        return self

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def is_active(self) -> bool:
        return self.status == TraineeStatus.ACTIVE

    @property
    def is_closed(self) -> bool:
        """Left or graduated: the file is closed, no new goals or progress."""
        return self.status in CLOSED_STATUSES

    def with_status(self, status: TraineeStatus, *, at: datetime | None = None) -> Self:
        """A NEW trainee with this status. self is unchanged (same idea as Goal.with_progress).
        Closing the file records when (at, default now); reopening it clears that. Moving between
        the two closed statuses keeps the original date: the file was closed once."""
        closing = status in CLOSED_STATUSES
        if closing and self.is_closed:
            closed_at = self.closed_at
        elif closing:
            closed_at = at or datetime.now(UTC)
        else:
            closed_at = None
        return self.model_validate({**self.model_dump(), "status": status, "closed_at": closed_at})


CLOSED_STATUSES = frozenset({TraineeStatus.LEFT, TraineeStatus.GRADUATED})


class PersonalFile(BaseModel):
    """Sensitive data. Visible to managers, or to counselors with approval."""

    model_config = ConfigDict(validate_assignment=True)

    id: int | None = None
    trainee_id: int
    intake_answers: dict[str, str] = Field(default_factory=dict, repr=False)
    background_notes: str | None = Field(default=None, repr=False)