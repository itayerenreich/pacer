"""Errors raised by the services layer.

The API (week 2, FastAPI) turns each one into an HTTP status:
    NotFoundError          -> 404
    PermissionDeniedError  -> 403
    RuleViolationError     -> 409   (also domain.DomainRuleError)
"""

from typing import TypeVar

T = TypeVar("T")


class ServiceError(Exception):
    """Base class: something the caller asked for cannot be done."""


class NotFoundError(ServiceError):
    """The object the caller referred to does not exist."""


class PermissionDeniedError(ServiceError):
    """The actor exists, but their role does not allow this action."""


class RuleViolationError(ServiceError):
    """The action breaks a business rule that involves more than one object."""


def found(value: T | None, what: str, object_id: int) -> T:
    """Return value, or raise NotFoundError if a repository returned None."""
    if value is None:
        raise NotFoundError(f"{what} {object_id} does not exist")
    return value
