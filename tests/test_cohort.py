"""Tests for the Cohort and Group domain models."""

from datetime import date

import pytest
from pydantic import ValidationError

from pacer.domain import Cohort, Group


def make_cohort(**overrides) -> Cohort:
    data = {"name": "Mahzor 12", "start_date": date(2026, 9, 1), "end_date": date(2027, 6, 30)}
    data.update(overrides)
    return Cohort(**data)


def make_group(**overrides) -> Group:
    data = {"name": "Group A", "cohort_id": 1}
    data.update(overrides)
    return Group(**data)


# --- Cohort ---


def test_valid_cohort_is_created():
    cohort = make_cohort()
    assert cohort.name == "Mahzor 12"
    assert cohort.id is None


def test_cohort_ending_before_start_is_rejected():
    with pytest.raises(ValidationError, match="end_date must be after start_date"):
        make_cohort(end_date=date(2025, 1, 1))


def test_cohort_ending_on_start_date_is_rejected():
    with pytest.raises(ValidationError, match="end_date must be after start_date"):
        make_cohort(end_date=date(2026, 9, 1))


def test_cohort_dates_given_as_strings_are_converted():
    cohort = make_cohort(start_date="2026-09-01", end_date="2027-06-30")
    assert cohort.start_date == date(2026, 9, 1)
    assert cohort.end_date == date(2027, 6, 30)


# --- Group ---


def test_group_without_counselors_is_valid():
    group = make_group()
    assert group.counselor_ids == []


def test_group_with_three_counselors_is_rejected():
    with pytest.raises(ValidationError):
        make_group(counselor_ids=[1, 2, 3])


def test_group_with_duplicate_counselor_is_rejected():
    with pytest.raises(ValidationError, match="Counselor 4 appears more than once"):
        make_group(counselor_ids=[4, 4])


def test_groups_do_not_share_counselor_lists():
    first = make_group()
    second = make_group()
    first.counselor_ids.append(7)
    assert second.counselor_ids == []
