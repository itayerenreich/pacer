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
* For the most common of those words (COMMON_WORD_NAMES: "כבר" is כ + בר!) a lone word is
  still masked, but with a neutral [NAME_n] token that turns back into the WORD, not into a
  person: otherwise "שלחתי אותו כבר" came back from the model as "שלחתי אותו כיעל בר", a
  sentence wrongly tied to a real trainee (decision 2026-10-15). The exception is context: when
  exactly one person in it (a meeting's participants, a talk's trainee and counselor) has that
  name, the word is that person. A full name ("יעל בר") is always the person.
* A first name alone may fit two people (two trainees named יעל). Full names are resolved to
  the right person; an ambiguous first name gets a neutral token ([NAME_1]) that points to no
  one, and the result reports it so a person can resolve it.
* First name alone, last name alone, and full name all map to the same person's token. Each word
  of a two-word first or last name counts too ("שרה לאה": "שרה" alone is her).
* Spelling variants of the same name match: apostrophe or geresh (ג'ני, ג׳ני, ג’ני), hyphen or
  maqaf (בר-און, בר־און), and upper or lower case for names in Latin letters (Dana, dana, DANA).

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
# Names that are also everyday words. The test for this list is how common the WORD is in speech,
# not how common the name is ("רון" is not here: as a word it is rare). Add to it when a case shows up.
COMMON_WORD_NAMES = frozenset(
    {"בר", "טל", "שחר", "בן", "אור", "אלה", "נתן", "שלו", "שני", "רעות", "ארז", "אלון", "חן", "גל", "דור"}
)
_TOKEN = re.compile(r"\[(?:TRAINEE|STAFF)_\d+\]|\[NAME_\d+\]")
# Characters that are written in several ways, and the one form we compare by.
_VARIANTS = {"'": "'׳’`´", '"': '"״“”', "-": "-־–"}
_CANONICAL = {variant: canonical for canonical, variants in _VARIANTS.items() for variant in variants}


def _normalize(form: str) -> str:
    """One way to write a name: single spaces, canonical apostrophe/quote/hyphen, lower case."""
    return "".join(_CANONICAL.get(ch, ch) for ch in " ".join(form.split())).lower()


def _regex_for(form: str) -> str:
    """A normalized form as a regex that also matches its other spellings."""
    parts = []
    for ch in form:
        if ch == " ":
            parts.append(r"\s+")
        elif ch in _VARIANTS:
            parts.append(f"[{re.escape(_VARIANTS[ch])}]")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


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
    """Every known person, and every way their name can appear: full, first only, last only,
    and each word of a name that has several."""

    def __init__(self, people: Iterable[Person]) -> None:
        self.people = tuple(people)
        forms: dict[str, set[Person]] = defaultdict(set)
        for person in self.people:
            for form in _forms_of(person):
                forms[_normalize(form)].add(person)
        forms.pop("", None)
        # A form fits one person, or several (ambiguous). Longest first, so a full name wins over
        # the first name inside it. Keys are normalized: look them up with _normalize(surface).
        self.forms: Mapping[str, frozenset[Person]] = MappingProxyType(
            {form: frozenset(people) for form, people in sorted(forms.items(), key=lambda kv: -len(kv[0]))}
        )
        alternatives = "|".join(_regex_for(form) for form in self.forms)
        self._by_key = {(person.kind, person.id): person for person in self.people}
        self.pattern = re.compile(
            rf"{_NOT_WORD_BEFORE}(?P<prefix>[{PREFIX_LETTERS}]{{0,3}}?)(?P<name>{alternatives}){_NOT_WORD_AFTER}"
            if alternatives
            else r"(?!)",  # an empty roster matches nothing
            re.IGNORECASE,  # Latin letters: Dana, dana, DANA. Hebrew has no case
        )

    def candidates(self, surface: str) -> frozenset[Person]:
        return self.forms[_normalize(surface)]


    def find(self, kind: PersonKind, person_id: int) -> Person | None:
        return self._by_key.get((kind, person_id))


class Anonymizer:
    def __init__(self, roster: Roster) -> None:
        self._roster = roster

    def anonymize(
        self, text: str, *, known: Iterable[Person] = (), prefer: Iterable[Person] = ()
    ) -> AnonymizedText:
        """known:  people the model is told about by token even if the text never names them
                   (e.g. the participants of a meeting), so their tokens in the answer map back.
        prefer: who the text is about. When a name fits several people and exactly one of them
                is preferred, it is that one (in a talk with Yael Mizrahi, "יעל" is her, not the
                other Yael). Otherwise an ambiguous name still gets a neutral [NAME_n] token.
        A name that is also a common word (COMMON_WORD_NAMES) is a person only when exactly one
        person in the context (known + prefer) has it; otherwise it gets a neutral [NAME_n]."""
        people: dict[str, Person] = {person.token: person for person in known}
        preferred = frozenset(prefer)
        context = preferred | frozenset(people.values())
        unresolved: dict[str, str] = {}
        ambiguous_tokens: dict[str, str] = {}  # the same ambiguous word gets the same token

        def replace(match: re.Match) -> str:
            surface = " ".join(match["name"].split())
            candidates = self._roster.candidates(surface)
            if _normalize(surface) in COMMON_WORD_NAMES:
                # A word as much as a name: it is a person only if the context says which one.
                candidates = candidates & context
                if len(candidates) != 1:
                    candidates = frozenset()  # -> a neutral token that restores to the word
            elif len(candidates) > 1 and len(candidates & preferred) == 1:
                candidates = candidates & preferred
            if len(candidates) == 1:
                (person,) = candidates
                people[person.token] = person
                return match["prefix"] + person.token
            key = _normalize(surface)  # "Dana" and "dana" are the same ambiguous name
            if key not in ambiguous_tokens:
                ambiguous_tokens[key] = f"[NAME_{len(ambiguous_tokens) + 1}]"
                unresolved[ambiguous_tokens[key]] = surface  # restored as it was first written
            return match["prefix"] + ambiguous_tokens[key]

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


def _forms_of(person: Person) -> list[str]:
    """Full name, first name, last name, and each word of a first or last name with several."""
    forms = [person.full_name, person.first_name, person.last_name]
    for name in (person.first_name, person.last_name):
        words = name.split()
        if len(words) > 1:
            forms += [word for word in words if len(word) > 1]
    return forms


def _split_name(full_name: str) -> tuple[str, str]:
    """Staff have one full_name field: the first word is the first name, the rest the last name."""
    first, _, last = " ".join(full_name.split()).partition(" ")
    return first, last
