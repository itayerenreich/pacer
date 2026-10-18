"""The base of every stored domain object: an id that is None until the object is saved."""

from pydantic import BaseModel

from .errors import DomainRuleError


class Entity(BaseModel):
    """id is None for a new object, and set by the repository when it is saved.

    That is the honest type, but most code works on objects that came FROM a repository, where
    the id is always there. `key` says so: it is the id as a plain int, and fails loudly (instead
    of passing None along) if the object was never saved. So the type checker can prove that
    every id handed to a repository or a rule is a real one.
    """

    id: int | None = None

    @property
    def key(self) -> int:
        if self.id is None:
            raise DomainRuleError(f"This {type(self).__name__} was not saved yet: it has no id")
        return self.id


def present[T](value: T | None, what: str) -> T:
    """A field that is optional in general, but that this object's own rules guarantee here
    (e.g. a progress proposal always has a goal_id; an anonymized meeting always has its
    transcript). Says so to the type checker, and fails loudly if the rule was ever broken."""
    if value is None:
        raise DomainRuleError(f"Expected {what}, but it is missing")
    return value
