"""Tests for the role rules in services/permissions.py (pure functions)."""

import pytest
from fakes import World

from pacer.services.errors import PermissionDeniedError
from pacer.services.permissions import can_edit_goals, can_view_trainee, require


@pytest.fixture
def world() -> World:
    return World()


def test_manager_can_view_and_edit_everyone(world):
    assert can_view_trainee(world.manager, world.group_a)
    assert can_view_trainee(world.manager, world.group_b)
    assert can_edit_goals(world.manager, world.yael)


def test_primary_counselor_can_view_and_edit(world):
    assert can_view_trainee(world.noa, world.group_a)
    assert can_edit_goals(world.noa, world.yael)


def test_second_group_counselor_can_view_but_not_edit(world):
    assert can_view_trainee(world.omer, world.group_a)
    assert not can_edit_goals(world.omer, world.yael)


def test_counselor_of_another_group_can_do_neither(world):
    assert not can_view_trainee(world.dana, world.group_a)
    assert not can_edit_goals(world.dana, world.yael)


def test_require_raises_when_not_allowed():
    require(True, "do this")  # no error
    with pytest.raises(PermissionDeniedError, match="Not allowed to do that"):
        require(False, "do that")
