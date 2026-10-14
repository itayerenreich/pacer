"""Translating errors from the inner layers into HTTP status codes.

The services and the domain know nothing about HTTP. They raise plain Python
errors, and this module decides how each one looks to an HTTP client:

    NotAuthenticatedError      401  we don't know who you are
    PermissionDeniedError      403  we know who you are, and you may not do this
    NotFoundError              404  no such trainee / goal / request
    RuleViolationError         409  conflicts with the current state of other objects
    DomainRuleError            409  not allowed in the object's current state
    pydantic.ValidationError   422  invalid values (same code FastAPI uses for bad input)

Every error body has the same shape as FastAPI's own: {"detail": ...}.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from pacer.domain import DomainRuleError
from pacer.services import NotFoundError, PermissionDeniedError, RuleViolationError


class NotAuthenticatedError(Exception):
    """The request does not say who is making it, or names an unknown staff member."""


STATUS_BY_ERROR: dict[type[Exception], int] = {
    NotAuthenticatedError: status.HTTP_401_UNAUTHORIZED,
    PermissionDeniedError: status.HTTP_403_FORBIDDEN,
    NotFoundError: status.HTTP_404_NOT_FOUND,
    RuleViolationError: status.HTTP_409_CONFLICT,
    DomainRuleError: status.HTTP_409_CONFLICT,
}


def status_for(exc: Exception) -> int:
    """The status of the closest listed class, so subclasses (InvalidTransitionError) inherit it."""
    for cls in type(exc).__mro__:
        if cls in STATUS_BY_ERROR:
            return STATUS_BY_ERROR[cls]
    raise LookupError(f"No HTTP status for {type(exc).__name__}")


async def _error_response(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=status_for(exc), content={"detail": str(exc)})


async def _validation_error_response(request: Request, exc: ValidationError) -> JSONResponse:
    """A domain model rejected a value that got past the request schema."""
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": errors})


def register_error_handlers(app: FastAPI) -> None:
    for error_class in STATUS_BY_ERROR:
        app.add_exception_handler(error_class, _error_response)
    app.add_exception_handler(ValidationError, _validation_error_response)
