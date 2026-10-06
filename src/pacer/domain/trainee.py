"""Trainees and their sensitive personal files."""

from pydantic import BaseModel, ConfigDict, Field

from .enums import TraineeStatus


class Trainee(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: int | None = None
    first_name: str = Field(min_length=1, max_length=50)
    last_name: str = Field(min_length=1, max_length=50)
    cohort_id: int
    group_id: int
    primary_counselor_id: int
    status: TraineeStatus = TraineeStatus.ACTIVE

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"

    @property
    def is_active(self) -> bool:
        return self.status == TraineeStatus.ACTIVE


class PersonalFile(BaseModel):
    """Sensitive data. Visible to managers, or to counselors with approval."""

    model_config = ConfigDict(validate_assignment=True)

    id: int | None = None
    trainee_id: int
    intake_answers: dict[str, str] = Field(default_factory=dict, repr=False)
    background_notes: str | None = Field(default=None, repr=False)