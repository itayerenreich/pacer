"""Request bodies: what a client may send, which is less than the domain objects hold.

A client creating a goal sends a title and a category. It does not choose the id,
created_at, status or progress, and the trainee comes from the URL. So each body
here is a separate small model, and the endpoint builds the domain object from it.

The field rules are not written twice: `Goal.model_fields["title"]` reuses the
domain's own rule (3-200 characters), so /docs shows it and it can never drift.
Responses are the domain models themselves (Trainee, Goal, ...), since everything
in them may be shown to a staff member who passed the permission check.
"""

from datetime import date
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from pacer.domain import (
    ApprovalKind,
    ApprovalRequest,
    Goal,
    GoalCategory,
    StaffMeeting,
    Trainee,
    TraineeStatus,
)


class _Body(BaseModel):
    model_config = ConfigDict(extra="forbid")  # an unknown field is a client bug: reject it


class TraineeCreate(_Body):
    first_name: str = Trainee.model_fields["first_name"]
    last_name: str = Trainee.model_fields["last_name"]
    cohort_id: int
    group_id: int
    primary_counselor_id: int


class GoalCreate(_Body):
    title: str = Goal.model_fields["title"]
    description: str | None = None
    category: GoalCategory
    target_date: date | None = None


class ProgressUpdate(_Body):
    # Same range as Goal.progress (written here because the domain field has a default
    # of 0, and here progress is required). The domain checks it again anyway.
    progress: Annotated[int, Field(ge=0, le=100, examples=[60])]
    note: Annotated[str, Field(min_length=1, max_length=1000, examples=["Ran 3 km without stopping"])]


class StatusChange(_Body):
    status: TraineeStatus


class ApprovalCreate(_Body):
    kind: ApprovalKind
    trainee_id: int
    reason: str = ApprovalRequest.model_fields["reason"]
    requested_status: TraineeStatus | None = Field(
        default=None, description="Required for kind=status_change, otherwise leave empty"
    )


class Decision(_Body):
    note: str | None = ApprovalRequest.model_fields["decision_note"]


class MeetingCreate(_Body):
    title: str = StaffMeeting.model_fields["title"]
    held_on: date
    participant_ids: list[int] = Field(min_length=1, description="Staff members who took part")
    transcript: str | None = Field(
        default=None,
        min_length=20,
        description="The meeting as text. Leave empty when an audio file will be uploaded instead (task 6).",
    )
