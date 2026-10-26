"""The weekly schedule, and who attended each slot of it.

Two levels (decision 2026-10-26, Itay):

    ScheduleSlot      the TEMPLATE: what a typical week looks like ("Sunday 07:00 morning sport").
                      Defined once. Publishing a week creates that week's sessions from it.
    ScheduledSession  one slot on one DATE ("Sunday 2026-10-25, 07:00 morning sport"). Attendance
                      is recorded on these. A week can be changed (a session moved, cancelled, a
                      one-off added) without touching the template, and changing the template never
                      rewrites a week that already happened.

A slot is for the whole cohort (group_id None) or for one group (a group talk).

Attendance has three statuses (decision 2026-10-26): present, absent, or excused. The rate is
present out of (present + absent): an excused absence does not count against the trainee. There is
no free-text reason, on purpose: "he was at the doctor" is health data, and what is never written
can never leak. A slot nobody marked yet is reported as unmarked, never as present or absent.

The academy's week starts on Sunday.
"""

from datetime import UTC, date, datetime, time, timedelta
from enum import IntEnum, StrEnum
from typing import Self

from pydantic import AwareDatetime, ConfigDict, Field, model_validator

from .entity import Entity
from .errors import DomainRuleError

TITLE_MAX_LENGTH = 100


class Weekday(IntEnum):
    """Python's numbering (date.weekday()): Monday is 0. Shown in Hebrew, in the academy's order."""

    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6

    @property
    def label(self) -> str:
        return _DAY_LABELS[self]

    @property
    def offset(self) -> int:
        """Days after the week's Sunday: Sunday 0, Monday 1 ... Saturday 6."""
        return (self.value + 1) % 7


_DAY_LABELS = {
    Weekday.SUNDAY: "ראשון",
    Weekday.MONDAY: "שני",
    Weekday.TUESDAY: "שלישי",
    Weekday.WEDNESDAY: "רביעי",
    Weekday.THURSDAY: "חמישי",
    Weekday.FRIDAY: "שישי",
    Weekday.SATURDAY: "שבת",
}


def week_start(day: date) -> date:
    """The Sunday that starts the academy week of this day."""
    return day - timedelta(days=Weekday(day.weekday()).offset)


class AttendanceStatus(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    EXCUSED = "excused"  # an absence that does not count against the trainee

    @property
    def label(self) -> str:
        return {"present": "הגיע", "absent": "לא הגיע", "excused": "היעדרות מוצדקת"}[self.value]


def _check_times(start: time, end: time) -> None:
    if end <= start:
        raise ValueError("A slot must end after it starts")


class ScheduleSlot(Entity):
    """A slot of the weekly template."""

    model_config = ConfigDict(validate_assignment=True)

    cohort_id: int
    group_id: int | None = None  # None: the whole cohort
    weekday: Weekday
    start: time
    end: time
    title: str = Field(min_length=2, max_length=TITLE_MAX_LENGTH)
    active: bool = True  # an inactive slot is no longer published; the weeks it made stay

    @model_validator(mode="after")
    def check_times(self) -> Self:
        _check_times(self.start, self.end)
        return self

    def on_week(self, sunday: date) -> date:
        """The date of this slot in the week that starts on `sunday`."""
        if Weekday(sunday.weekday()) is not Weekday.SUNDAY:
            raise DomainRuleError(f"A week starts on Sunday; {sunday} is not one")
        return sunday + timedelta(days=self.weekday.offset)


class ScheduledSession(Entity):
    """One slot on one date. Attendance is recorded here."""

    model_config = ConfigDict(validate_assignment=True)

    cohort_id: int
    group_id: int | None = None
    on: date
    start: time
    end: time
    title: str = Field(min_length=2, max_length=TITLE_MAX_LENGTH)
    slot_id: int | None = None  # the template slot it came from; None for a one-off
    cancelled: bool = False  # a cancelled session takes no attendance and is not counted

    @model_validator(mode="after")
    def check_times(self) -> Self:
        _check_times(self.start, self.end)
        return self

    @classmethod
    def from_slot(cls, slot: ScheduleSlot, sunday: date) -> Self:
        return cls(
            cohort_id=slot.cohort_id,
            group_id=slot.group_id,
            on=slot.on_week(sunday),
            start=slot.start,
            end=slot.end,
            title=slot.title,
            slot_id=slot.key,
        )

    def changed(self, **changes) -> Self:
        """A NEW session with these changes (title, start, end, on), fully validated."""
        if self.cancelled:
            raise DomainRuleError("A cancelled session cannot be changed")
        if unknown := set(changes) - {"title", "start", "end", "on"}:
            raise DomainRuleError(f"Cannot change {sorted(unknown)} of a session")
        return self.model_validate({**self.model_dump(), **changes})

    def cancel(self) -> Self:
        if self.cancelled:
            raise DomainRuleError("The session is already cancelled")
        return self.model_validate({**self.model_dump(), "cancelled": True})


class AttendanceRecord(Entity):
    """Whether one trainee attended one session. One per (session, trainee): marking again replaces it."""

    model_config = ConfigDict(validate_assignment=True)

    session_id: int
    trainee_id: int
    status: AttendanceStatus
    recorded_by_id: int
    recorded_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
