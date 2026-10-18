"""Goals that trainees set with their counselor, and their progress history."""

from datetime import UTC, date, datetime
from typing import Self

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from .entity import Entity
from .enums import GoalCategory, GoalStatus
from .errors import DomainRuleError


class Goal(Entity):
    model_config = ConfigDict(validate_assignment=True)

    trainee_id: int
    title: str = Field(min_length=3, max_length=200)
    description: str | None = None
    category: GoalCategory
    status: GoalStatus = GoalStatus.NOT_STARTED
    progress: int = Field(default=0, ge=0, le=100)
    target_date: date | None = None
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def check_achieved_is_complete(self) -> Self:
        if self.status == GoalStatus.ACHIEVED and self.progress != 100:
            raise ValueError("An achieved goal must have progress 100")
        return self

    def with_progress(self, progress: int) -> Self:
        """A NEW goal at this progress, with the status that matches it. self is unchanged.

        Building a new object (instead of assigning goal.progress = ...) means every
        rule is checked on the complete new state before anyone can see it. This is
        how services avoid the known validate_assignment bug (see test_goal.py).
        """
        if self.status == GoalStatus.DROPPED:
            raise DomainRuleError("A dropped goal cannot get progress updates")
        if progress == 100:
            status = GoalStatus.ACHIEVED
        elif progress > 0:
            status = GoalStatus.IN_PROGRESS
        else:
            status = GoalStatus.NOT_STARTED
        return self.model_validate({**self.model_dump(), "progress": progress, "status": status})


class GoalUpdate(Entity):
    model_config = ConfigDict(validate_assignment=True)

    goal_id: int
    author_id: int
    progress: int = Field(ge=0, le=100)
    note: str = Field(min_length=1, max_length=1000)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
