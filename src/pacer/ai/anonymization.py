"""Anonymization: replace the names of people we know with stable tokens, in code, before any
text goes to a model that does not run on this computer.

    anonymizer = Anonymizer(roster)                  # roster: every trainee and staff member
    safe = anonymizer.anonymize("נועה תדבר עם ליעל על הריצה")
    safe.text      ->  "[STAFF_2] תדבר עם ל[TRAINEE_7] על הריצה"
    ... the model works on safe.text ...
    deanonymize(model_answer, safe)                  # tokens back to names, on this computer

Why code and not a model: we already know every name (they are in the database), so matching
them is exact, fast, free and fully testable. A small local model would be slower and would
"guess"; a cloud model would see the names it is meant to hide.

Hebrew makes this harder than it looks:
* Prefix letters (ו ה ב כ ל מ ש, up to three together) attach to names: "ליעל", "ושחר",
  "כשמאיה". The prefix stays, the name becomes a token: "ל[TRAINEE_7]".
* Many names are also ordinary words: טל (dew), שחר (dawn), אגם (lake), בר. The policy is
  OVER-MASKING: when in doubt, replace. "בשחר" (at dawn) becomes "ב[TRAINEE_3]"; a slightly
  odd sentence for the model is far better than a name leaving the computer.
* A first name alone may fit two people (two trainees named יעל). Full names are resolved to
  the right person; an ambiguous first name gets a neutral token ([NAME_1]) that points to no
  one, and the result reports it so a person can resolve it.
* First name alone, last name alone, and full name all map to the same person's token.

Known limit: names that are NOT in the roster (a parent, someone outside the academy) are not
detected. The prompts tell the model to use roles, not names, and the plan tracks this risk.

Tokens use database ids ([TRAINEE_7], [STAFF_2]). An id is not personal information, and it
lets later stages map a model's answer ("owner: [STAFF_2]") straight back to a staff member.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from pacer.repositories.interfaces import StaffRepo, TraineeRepo

PREFIX_LETTERS = "ובכלמשה"
_NOT_WORD_BEFORE = r"(?<!\w)"  # \w includes Hebrew letters in Python 3 regexes
_NOT_WORD_AFTER = r"(?!\w)"
_TOKEN = re.compile(r"\[(?:TRAINEE|STAFF)_\d+\]|\[NAME_\d+\]")


class PersonKind(StrEnum):
    TRAINEE = "TRAINEE"
    STAFF = "STAFF"


@dataclass(frozen=True)
class Person:
    kind: PersonKind
    id: int
    first_name: str
    last_name: str

    @property
    def token(self) -> str:
        return f"[{self.kind}_{self.id}]"

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class AnonymizationError(Exception):
    """Anonymization could not guarantee its result. Nothing may be sent."""


_CREATED_BY_ANONYMIZER = object()


@dataclass(frozen=True)
class AnonymizedText:
    """Text that went through Anonymizer.anonymize(). Only the Anonymizer can create one.

    Pipeline stages that send text to a cloud model accept AnonymizedText, not str, so raw
    text cannot be passed to them by mistake: the type itself is the proof of anonymization.
    """

    text: str
    people: Mapping[str, Person] = field(repr=False)  # token -> the person it stands for
    unresolved: Mapping[str, str] = field(repr=False)  # [NAME_n] -> the ambiguous word in the original
    _key: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._key is not _CREATED_BY_ANONYMIZER:
            raise TypeError("AnonymizedText is created only by Anonymizer.anonymize()")

    def __repr__(self) -> str:  # never show names, even in a debugger or an error message
        return f"AnonymizedText({len(self.text)} chars, {len(self.people)} people, {len(self.unresolved)} unresolved)"

    def person_for(self, token: str) -> Person | None:
        return self.people.get(token)


class Roster:
    """Every known person, and every way their name can appear: full, first only, last only."""

    def __init__(self, people: Iterable[Person]) -> None:
        self.people = tuple(people)
        forms: dict[str, set[Person]] = defaultdict(set)
        for person in self.people:
            for form in (person.full_name, person.first_name, person.last_name):
                form = " ".join(form.split())
                if form:
                    forms[form].add(person)
        # A surface form fits one person, or several (ambiguous). Longest first, so a full name
        # wins over the first name inside it.
        self.forms: Mapping[str, frozenset[Person]] = MappingProxyType(
            {form: frozenset(people) for form, people in sorted(forms.items(), key=lambda kv: -len(kv[0]))}
        )
        alternatives = "|".join(r"\s+".join(map(re.escape, form.split())) for form in self.forms)
        self._by_key = {(person.kind, person.id): person for person in self.people}
        self.pattern = re.compile(
            rf"{_NOT_WORD_BEFORE}(?P<prefix>[{PREFIX_LETTERS}]{{0,3}}?)(?P<name>{alternatives}){_NOT_WORD_AFTER}"
            if alternatives
            else r"(?!)"  # an empty roster matches nothing
        )


    def find(self, kind: PersonKind, person_id: int) -> Person | None:
        return self._by_key.get((kind, person_id))


class Anonymizer:
    def __init__(self, roster: Roster) -> None:
        self._roster = roster

    def anonymize(self, text: str, *, known: Iterable[Person] = ()) -> AnonymizedText:
        """known: people the model is told about by token even if the text never names them
        (e.g. the participants of a meeting), so their tokens in the answer can be mapped back."""
        people: dict[str, Person] = {person.token: person for person in known}
        unresolved: dict[str, str] = {}
        ambiguous_tokens: dict[str, str] = {}  # the same ambiguous word gets the same token

        def replace(match: re.Match) -> str:
            surface = " ".join(match["name"].split())
            candidates = self._roster.forms[surface]
            if len(candidates) == 1:
                (person,) = candidates
                people[person.token] = person
                return match["prefix"] + person.token
            if surface not in ambiguous_tokens:
                ambiguous_tokens[surface] = f"[NAME_{len(ambiguous_tokens) + 1}]"
                unresolved[ambiguous_tokens[surface]] = surface
            return match["prefix"] + ambiguous_tokens[surface]

        result = self._roster.pattern.sub(replace, text)
        if self._roster.pattern.search(result):  # defense in depth: should never happen
            raise AnonymizationError("A known name is still present after anonymization")
        return AnonymizedText(
            text=result,
            people=MappingProxyType(people),
            unresolved=MappingProxyType(unresolved),
            _key=_CREATED_BY_ANONYMIZER,
        )

    def contains_known_name(self, text: str) -> bool:
        return self._roster.pattern.search(text) is not None


def deanonymize(text: str, source: AnonymizedText) -> str:
    """Tokens back to names, on this computer. Unknown tokens (invented by a model) stay as they
    are, so a later validation step can notice them instead of silently guessing."""

    def restore(match: re.Match) -> str:
        token = match.group(0)
        if person := source.people.get(token):
            return person.full_name
        return source.unresolved.get(token, token)

    return _TOKEN.sub(restore, text)


def tokens_in(text: str) -> list[str]:
    """Every token in a text, in order. Used to validate a model's answer."""
    return _TOKEN.findall(text)


def load_roster(staff: StaffRepo, trainees: TraineeRepo) -> Roster:
    """Everyone whose name may appear: all staff, and all trainees in every status (a trainee
    who left is still a person whose name must not leave the computer)."""
    people = [Person(PersonKind.STAFF, s.id, *_split_name(s.full_name)) for s in staff.list_all()]
    people += [Person(PersonKind.TRAINEE, t.id, t.first_name, t.last_name) for t in trainees.list_all()]
    return Roster(people)


def _split_name(full_name: str) -> tuple[str, str]:
    """Staff have one full_name field: the first word is the first name, the rest the last name."""
    first, _, last = " ".join(full_name.split()).partition(" ")
    return first, last
