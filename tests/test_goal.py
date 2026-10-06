"""Tests for the Goal domain model."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from pacer.domain import Goal, GoalStatus


def make_goal(**overrides) -> Goal:
    data = {"trainee_id": 1, "title": "Run 5 km", "category": "physical"}
    data.update(overrides)
    return Goal(**data)


def test_new_goal_has_defaults():
    goal = make_goal()
    assert goal.status == GoalStatus.NOT_STARTED
    assert goal.progress == 0


def test_achieved_goal_at_100_is_valid():
    goal = make_goal(status="achieved", progress=100)
    assert goal.status == GoalStatus.ACHIEVED


def test_achieved_goal_below_100_is_rejected():
    with pytest.raises(ValidationError, match="achieved goal must have progress 100"):
        make_goal(status="achieved", progress=60)


def test_in_progress_goal_can_be_partial():
    goal = make_goal(status="in_progress", progress=60)
    assert goal.progress == 60


@pytest.mark.parametrize("progress", [-1, 101, 150])
def test_progress_out_of_range_is_rejected(progress):
    with pytest.raises(ValidationError):
        make_goal(progress=progress)


def test_invalid_assignment_is_rejected_and_rolled_back():
    goal = make_goal()
    with pytest.raises(ValidationError):
        goal.progress = 500
    assert goal.progress == 0


@pytest.mark.xfail(strict=True, reason="Known bug: a failed model_validator on assignment keeps the new value")
def test_failed_model_rule_on_assignment_keeps_old_value():
    goal = make_goal(progress=60)
    with pytest.raises(ValidationError):
        goal.status = "achieved"
    assert goal.status == GoalStatus.NOT_STARTED


def test_created_at_is_timezone_aware_utc():
    goal = make_goal()
    assert goal.created_at.utcoffset() is not None
    assert goal.created_at.utcoffset().total_seconds() == 0


def test_naive_created_at_is_rejected():
    with pytest.raises(ValidationError):
        make_goal(created_at=datetime(2026, 10, 6, 12, 0))
