"""Cohorts (annual intakes) and the groups inside them."""

from datetime import date
from typing import Self

from pydantic import BaseModel, Field, model_validator


class Cohort(BaseModel):
    id: int | None = None
    name: str = Field(min_length=1, max_length=50)
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def check_dates(self) -> Self:
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self


class Group(BaseModel):
    id: int | None = None
    name: str = Field(min_length=1, max_length=50)
    cohort_id: int
    counselor_ids: list[int] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def check_unique_counselors(self) -> Self:
        seen = set()
        for counselor_id in self.counselor_ids:
            if counselor_id in seen:
                raise ValueError(f"Counselor {counselor_id} appears more than once")
            seen.add(counselor_id)
        return self
