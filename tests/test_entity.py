"""Entity.key and present(): the places where "optional in general" meets "guaranteed here"."""

import pytest

from pacer.domain import StaffMember, StaffRole, present
from pacer.domain.errors import DomainRuleError


def test_a_saved_entity_gives_its_id_as_a_plain_int():
    assert StaffMember(id=3, full_name="נועה לוי", role=StaffRole.COUNSELOR).key == 3


def test_an_unsaved_entity_fails_loudly_instead_of_passing_none_along():
    with pytest.raises(DomainRuleError, match="StaffMember was not saved yet"):
        _ = StaffMember(full_name="נועה לוי", role=StaffRole.COUNSELOR).key


def test_present_returns_the_value_or_says_what_is_missing():
    assert present(0, "a number") == 0  # falsy is not missing
    with pytest.raises(DomainRuleError, match="Expected a goal"):
        present(None, "a goal")
