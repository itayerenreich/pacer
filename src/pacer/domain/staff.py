"""Staff members: counselors and the manager."""

from pydantic import ConfigDict, Field

from .entity import Entity
from .enums import ResponsibilityArea, StaffRole


class StaffMember(Entity):
    model_config = ConfigDict(validate_assignment=True)

    full_name: str = Field(min_length=2, max_length=100)
    role: StaffRole
    responsibility_area: ResponsibilityArea | None = None
