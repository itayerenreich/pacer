"""Translating errors from the inner layers into HTTP status codes.

The services and the domain know nothing about HTTP. They raise plain Python
errors, and this module decides how each one looks to an HTTP client:

    NotAuthenticatedError      401  we don't know who you are
    PermissionDeniedError      403  we know who you are, and you may not do this
    NotFoundError              404  no such trainee / goal / request
    RuleViolationError         409  conflicts with the current state of other objects
    DomainRuleError            409  not allowed in the object's current state
    pydantic.ValidationError   422  invalid values (same code FastAPI uses for bad input)

AI steps (stage 3):
    StructuredOutputError      502  the model answered, but not validly, twice. Its answer was not saved
    LLMError (and subclasses)  503  the model could not be used: no key, down, rate-limited
    PrivacyViolation,          500  a safety check stopped the request. A bug, never "normal":
    AnonymizationError,             logged as an ERROR
    RoutingError

Every error body has the same shape as FastAPI's own: {"detail": ...}. For AI errors the detail
is a fixed sentence, not the error text: a provider's message is not ours to forward.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError

from pacer.ai import LLMError
from pacer.ai.anonymization import AnonymizationError
from pacer.ai.routing import PrivacyViolation, RoutingError
from pacer.ai.structured import StructuredOutputError
from pacer.domain import DomainRuleError
from pacer.services import NotFoundError, PermissionDeniedError, RuleViolationError

log = logging.getLogger(__name__)


class NotAuthenticatedError(Exception):
    """The request does not say who is making it, or names an unknown staff member."""


STATUS_BY_ERROR: dict[type[Exception], int] = {
    NotAuthenticatedError: status.HTTP_401_UNAUTHORIZED,
    PermissionDeniedError: status.HTTP_403_FORBIDDEN,
    NotFoundError: status.HTTP_404_NOT_FOUND,
    RuleViolationError: status.HTTP_409_CONFLICT,
    DomainRuleError: status.HTTP_409_CONFLICT,
    # The database's own rules (unique, check): a rule the services did not see coming, e.g. two
    # requests at the same moment. A conflict, not a crash; the message never shows the SQL.
    IntegrityError: status.HTTP_409_CONFLICT,
    StructuredOutputError: status.HTTP_502_BAD_GATEWAY,
    LLMError: status.HTTP_503_SERVICE_UNAVAILABLE,
    PrivacyViolation: status.HTTP_500_INTERNAL_SERVER_ERROR,
    AnonymizationError: status.HTTP_500_INTERNAL_SERVER_ERROR,
    RoutingError: status.HTTP_500_INTERNAL_SERVER_ERROR,
}

_FIXED_DETAIL: dict[type[Exception], str] = {
    StructuredOutputError: "The model's answer was invalid twice and was not saved. Run it again.",
    LLMError: "The language model could not be used right now. Stages already done are kept; try again later.",
    PrivacyViolation: "A privacy check stopped this request. Nothing was sent or saved.",
    AnonymizationError: "A privacy check stopped this request. Nothing was sent or saved.",
    RoutingError: "The AI configuration is invalid. Nothing was sent or saved.",
    IntegrityError: "This change conflicts with the data as it is now (perhaps a change made at the same "
    "moment). Nothing was saved; reload and try again.",
}


def _closest[V](exc: Exception, table: dict[type[Exception], V]) -> V | None:
    """The value of the closest listed class, so subclasses (InvalidTransitionError) inherit it."""
    for cls in type(exc).__mro__:
        if cls in table:
            return table[cls]
    return None


def status_for(exc: Exception) -> int:
    code = _closest(exc, STATUS_BY_ERROR)
    if code is None:
        raise LookupError(f"No HTTP status for {type(exc).__name__}")
    return code


async def _error_response(request: Request, exc: Exception) -> JSONResponse:
    code = status_for(exc)
    detail = _closest(exc, _FIXED_DETAIL) or str(exc)
    if code >= 500:  # our side failed: always worth a look. The class name only, never the text
        log.error("%s %s -> %s %s", request.method, request.url.path, code, type(exc).__name__)
    else:  # 401/403 are worth noticing (someone tried something they may not do); the rest are routine
        level = logging.WARNING if code in (401, 403) else logging.INFO
        log.log(level, "%s %s -> %s %s: %s", request.method, request.url.path, code, type(exc).__name__, exc)
    return JSONResponse(status_code=code, content={"detail": detail})


async def _validation_error_response(request: Request, exc: Exception) -> JSONResponse:
    """A domain model rejected a value that got past the request schema."""
    if not isinstance(exc, ValidationError):  # registered for ValidationError only; Starlette types it wider
        raise TypeError(f"Expected a ValidationError, got {type(exc).__name__}")
    errors = exc.errors(include_url=False, include_context=False, include_input=False)
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, content={"detail": errors})


def register_error_handlers(app: FastAPI) -> None:
    for error_class in STATUS_BY_ERROR:
        app.add_exception_handler(error_class, _error_response)
    app.add_exception_handler(ValidationError, _validation_error_response)
