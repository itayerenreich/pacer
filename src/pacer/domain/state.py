"""One helper for every state machine in the domain: is this move allowed?"""

from collections.abc import Mapping
from enum import StrEnum

from .errors import InvalidTransitionError


def require_transition[S: StrEnum](
    transitions: Mapping[S, frozenset[S]], current: S, new: S, what: str
) -> None:
    if new not in transitions[current]:
        if not transitions[current]:
            raise InvalidTransitionError(f"This {what} is already {current}")
        raise InvalidTransitionError(f"Cannot move a {what} from {current} to {new}")
