"""Staff members: counselors and the manager."""

from pydantic import BaseModel, Field

from .enums import ResponsibilityArea, StaffRole


class StaffMember(BaseModel):
    id: int | None = None
    full_name: str = Field(min_length=2, max_length=100)
    role: StaffRole
    responsibility_area: ResponsibilityArea | None = None