"""Errors raised by domain rules that are about one object's behavior, not its fields.

Invalid field values raise pydantic.ValidationError. These errors are for actions
that are not allowed in the object's current state, e.g. approving a request twice.
"""


class DomainRuleError(Exception):
    """An action is not allowed in the object's current state."""


class InvalidTransitionError(DomainRuleError):
    """A state machine was asked to move along a transition it does not have."""
