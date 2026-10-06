"""Goals that trainees set with their counselor, and their progress history."""

from datetime import UTC, date, datetime
from typing import Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from .enums import GoalCategory, GoalStatus


class Goal(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: int | None = None
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


class GoalUpdate(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: int | None = None
    goal_id: int
    author_id: int
    progress: int = Field(ge=0, le=100)
    note: str = Field(min_length=1, max_length=1000)
    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
