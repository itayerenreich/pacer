"""Fixed sets of values used across the domain."""

from enum import StrEnum


class StaffRole(StrEnum):
    COUNSELOR = "counselor"
    MANAGER = "manager"


class ResponsibilityArea(StrEnum):
    VOLUNTEERING = "volunteering"
    LOGISTICS = "logistics"
    KITCHEN = "kitchen"
    CONTENT = "content"
    ARMY_PREP = "army_prep"


class TraineeStatus(StrEnum):
    ACTIVE = "active"
    ON_LEAVE = "on_leave"
    LEFT = "left"
    GRADUATED = "graduated"


class GoalCategory(StrEnum):
    PERSONAL = "personal"
    SOCIAL = "social"
    VALUES = "values"
    PHYSICAL = "physical"
    LEADERSHIP = "leadership"
    ARMY_PREP = "army_prep"


class GoalStatus(StrEnum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    DROPPED = "dropped"