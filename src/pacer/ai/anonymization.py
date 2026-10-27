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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from pacer.repositories.interfaces import StaffRepo, TraineeRepo

PREFIX_LETTERS = "ובכלמשה"
# A name ends where LETTERS end. Not \w: that includes digits and "_", so "מאיה5" or "dana_" would
# hide a name from the match. [^\W\d_] is "a letter" (Hebrew or Latin) in Python 3 regexes.
_LETTER = r"[^\W\d_]"
_NOT_LETTER_BEFORE = rf"(?<!{_LETTER})"
_NOT_LETTER_AFTER = rf"(?!{_LETTER})"
# Invisible characters that split a name without changing how it looks: niqqud and cantillation,
# zero-width characters, the soft hyphen, and direction marks. Removed from the text first.
# NOT the Hebrew punctuation that sits in the same Unicode block: the maqaf (U+05BE) and paseq, sof
# pasuq and nun hafukha (U+05C0, U+05C3, U+05C6) separate WORDS. Removed, "עם־נדב" became one word,
# "עםנדב", and the name went to the cloud (found by an independent review, 2026-10-25).
_INVISIBLE = re.compile(
    "[\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7\u00ad\u061c\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]"
)
# Separators inside a name: spaces and every kind of hyphen and dash. "בר-און", "בר און", "בר־און",
# "בר‑און" and even "בראון" are the same last name.
_SEPARATORS = " -\u05be\u2010\u2011\u2012\u2013\u2014\u2212"
_SEPARATOR = re.compile(f"[\\s{re.escape(_SEPARATORS[1:])}]+")
# Names that are also everyday words. The test for this list is how common the WORD is in speech,
# not how common the name is ("רון" is not here: as a word it is rare). Add to it when a case shows up.
COMMON_WORD_NAMES = frozenset(
    {"בר", "טל", "שחר", "בן", "אור", "אלה", "נתן", "שלו", "שני", "רעות", "ארז", "אלון", "חן", "גל", "דור"}
)
# Where a common-word name is always the word, whoever is in the context: "יום שני" is Monday.
_ALWAYS_THE_WORD_AFTER = {"שני": re.compile(r"יום\s*$")}
_TOKEN = re.compile(r"\[(?:TRAINEE|STAFF)_\d+\]|\[NAME_\d+\]")
# A word for the independent check: letters, with apostrophes and quotes inside (ג'ני, צה"ל).
_WORD = re.compile(r"[^\W\d_](?:[^\W\d_]|['׳’`´\"״“”])*")
# Characters that are written in several ways, and the one form we compare by.
_VARIANTS = {"'": "'׳’`´", '"': '"״“”'}
_CANONICAL = {variant: canonical for canonical, variants in _VARIANTS.items() for variant in variants}
# Where a span a person marked as a word waits while names are replaced: private-use characters and a
# number. No letters in it, so no name can match it or across it.
_WORD_SLOT = re.compile("\ue000(\\d+)\ue001")


def _clean(text: str) -> str:
    """The text without invisible characters. Done before matching AND before sending."""
    return _INVISIBLE.sub("", text)


def _normalize(form: str) -> str:
    """One way to write a name, to compare by: no invisible characters, no separators at all
    (spaces and hyphens), canonical apostrophe and quote, case-folded (Dana, dana, DANA)."""
    return _SEPARATOR.sub("", "".join(_CANONICAL.get(ch, ch) for ch in _clean(form))).casefold()


def _parts(name: str) -> list[str]:
    """The words of a name, split on spaces AND hyphens: "בר-און" -> ["בר", "און"]."""
    return [part for part in _SEPARATOR.split(_clean(name)) if part]


def _regex_for(form: str) -> str:
    """A form as a regex that matches its other spellings: any separator (or none) between its
    words, and every way of writing an apostrophe or a quote."""
    words = []
    for word in _parts(form):
        canonical = (_CANONICAL.get(c, c) for c in word)
        words.append("".join(f"[{re.escape(_VARIANTS[c])}]" if c in _VARIANTS else re.escape(c) for c in canonical))
    return rf"[\s{re.escape(_SEPARATORS[1:])}]*".join(words)


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
        # Every single word of every name, normalized, for the independent check
        # (contains_known_name): it does not use the big regex, so a bug in one is caught by the other.
        self._words = frozenset(
            {_normalize(part) for person in self.people for form in _forms_of(person) for part in _parts(form)}
            | set(forms)
        )
        # A form fits one person, or several (ambiguous). Longest first, so a full name wins over
        # the first name inside it. Keys are normalized: look them up with _normalize(surface).
        self.forms: Mapping[str, frozenset[Person]] = MappingProxyType(
            {form: frozenset(people) for form, people in sorted(forms.items(), key=lambda kv: -len(kv[0]))}
        )
        # The regex is built from the forms as WRITTEN (the keys have no separators left), longest first.
        written = {form for person in self.people for form in _forms_of(person) if _normalize(form)}
        alternatives = "|".join(_regex_for(form) for form in sorted(written, key=lambda f: -len(_normalize(f))))
        self._by_key = {(person.kind, person.id): person for person in self.people}
        # Any number of prefix letters ("וכשלמאיה" is ו+כש+ל), lazily: first try the whole word as a
        # name, so "מאור" is Maor, not מ + אור.
        self.pattern = re.compile(
            rf"{_NOT_LETTER_BEFORE}(?P<prefix>[{PREFIX_LETTERS}]*?)(?P<name>{alternatives}){_NOT_LETTER_AFTER}"
            if alternatives
            else r"(?!)",  # an empty roster matches nothing
            re.IGNORECASE,  # Latin letters: Dana, dana, DANA. Hebrew has no case
        )

    def candidates(self, surface: str) -> frozenset[Person]:
        found = self.forms.get(_normalize(surface))
        if found is None:  # the regex matched a spelling the lookup does not know: never guess
            raise AnonymizationError("A name matched, but could not be looked up; nothing may be sent")
        return found

    def has_name_in(self, text: str) -> bool:
        """Independent of `pattern`: split the text into words, peel any prefix letters off each,
        and look the rest up in the set of every name word. Tokens ([STAFF_3]) are not names."""
        for match in _WORD.finditer(_clean(_TOKEN.sub(" ", text))):
            word = _normalize(match.group())
            for start in range(len(word)):
                if word[start:] in self._words:
                    return True
                if word[start] not in PREFIX_LETTERS:
                    break
        return False


    def find(self, kind: PersonKind, person_id: int) -> Person | None:
        return self._by_key.get((kind, person_id))


class Anonymizer:
    def __init__(self, roster: Roster) -> None:
        self._roster = roster

    def anonymize(
        self,
        text: str,
        *,
        known: Iterable[Person] = (),
        prefer: Iterable[Person] = (),
        words: Sequence[tuple[int, int]] = (),
    ) -> AnonymizedText:
        """known:  people the model is told about by token even if the text never names them
                   (e.g. the participants of a meeting), so their tokens in the answer map back.
        prefer: who the text is about. When a name fits several people and exactly one of them
                is preferred, it is that one (in a talk with Yael Mizrahi, "יעל" is her, not the
                other Yael). Otherwise an ambiguous name still gets a neutral [NAME_n] token.
        A name that is also a common word (COMMON_WORD_NAMES) is a person only when exactly one
        person in the context (known + prefer) has it; otherwise it gets a neutral [NAME_n].
        words:  spans (start, end) of `text` that a PERSON marked as words, not names ("טיול לאגם",
                to the lake). They are left as written, and the final name check runs without them:
                a person decided. Nothing in code marks a span by itself (used by the evaluation, to
                measure what over-masking costs)."""
        people: dict[str, Person] = {person.token: person for person in known}
        preferred = frozenset(prefer)
        context = set(preferred) | set(people.values())  # grows with every name resolved in full
        unresolved: dict[str, str] = {}
        ambiguous_tokens: dict[str, str] = {}  # the same ambiguous word gets the same token
        text, marked_words = _set_aside(text, words)
        text = _clean(text)

        def resolve(match: re.Match) -> Person | None:
            surface, prefix = match["name"], match["prefix"]
            candidates = self._roster.candidates(surface)
            if _normalize(surface) in COMMON_WORD_NAMES:
                # As much a word as a name. It is a person only when the name fits exactly ONE person
                # in the whole roster, that person is in the context, it carries no prefix ("כבר" is
                # never כ + בר), and it is not a fixed phrase ("יום שני"). Context never breaks a tie.
                always_word = _ALWAYS_THE_WORD_AFTER.get(_normalize(surface))
                if (
                    prefix
                    or len(candidates) != 1
                    or not candidates <= context
                    or (always_word is not None and always_word.search(text, 0, match.start()))
                ):
                    return None
            elif len(candidates) > 1 and len(candidates & preferred) == 1:
                candidates = candidates & preferred
            if len(candidates) != 1:
                return None
            (person,) = candidates
            return person

        def replace(match: re.Match) -> str:
            person = resolve(match)
            if person is not None:
                people[person.token] = person
                if len(_parts(match["name"])) > 1:  # named in full: later, "שחר" alone may be them
                    context.add(person)
                return match["prefix"] + person.token
            surface = " ".join(match["name"].split())
            key = _normalize(surface)  # "Dana" and "dana" are the same ambiguous name
            if key not in ambiguous_tokens:
                ambiguous_tokens[key] = f"[NAME_{len(ambiguous_tokens) + 1}]"
                unresolved[ambiguous_tokens[key]] = surface  # restored as it was first written
            return match["prefix"] + ambiguous_tokens[key]

        result = self._roster.pattern.sub(replace, text)
        if self.contains_known_name(result):  # defense in depth, with an independent check too
            raise AnonymizationError("A known name is still present after anonymization")
        if marked_words:  # the marked words, back in their slots
            result = _WORD_SLOT.sub(lambda m: marked_words[int(m[1])], result)
        return AnonymizedText(
            text=result,
            people=MappingProxyType(people),
            unresolved=MappingProxyType(unresolved),
            _key=_CREATED_BY_ANONYMIZER,
        )

    def contains_known_name(self, text: str) -> bool:
        """Two different checks, so a bug in one is caught by the other: the matching regex, and a
        word-by-word lookup that does not use it (Roster.has_name_in)."""
        cleaned = _clean(text)
        return self._roster.pattern.search(cleaned) is not None or self._roster.has_name_in(cleaned)


def _set_aside(text: str, spans: Sequence[tuple[int, int]]) -> tuple[str, list[str]]:
    """The text with each marked span replaced by a numbered slot, and the spans' words in order."""
    if spans and _WORD_SLOT.search(text):  # a slot already in the text would take a marked word's place
        raise ValueError("The text contains the characters used to set words aside")
    pieces: list[str] = []
    marked: list[str] = []
    last = 0
    for start, end in sorted(spans):
        if not 0 <= last <= start < end <= len(text):
            raise ValueError(f"Marked words must be inside the text and not overlap: {start}-{end}")
        pieces += [text[last:start], f"\ue000{len(marked)}\ue001"]
        marked.append(text[start:end])
        last = end
    return "".join(pieces) + text[last:], marked


def deanonymize(text: str, source: AnonymizedText, name_for: Callable[[Person], str] | None = None) -> str:
    """Tokens back to names, on this computer. Unknown tokens (invented by a model) stay as they
    are, so a later validation step can notice them instead of silently guessing.
    name_for: how to write a person back (default: the full name). The coach uses it so that a
    trainee the asker may not see comes back only as the asker wrote them, never completed."""

    def restore(match: re.Match) -> str:
        token = match.group(0)
        if person := source.people.get(token):
            return person.full_name if name_for is None else name_for(person)
        return source.unresolved.get(token, token)

    return _TOKEN.sub(restore, text)


class AnonymizationSession:
    """One token space for a whole conversation (stage 6: the agent).

    The agent sends many texts to the model over one conversation: the counselor's message, earlier
    turns, and the result of every tool. A person's token is stable by itself ([TRAINEE_7] is trainee 7
    everywhere), but [NAME_n] is numbered per text: anonymized one by one, two texts would each have a
    [NAME_1] for two different words, and the model would take them for one person. The session renumbers
    every text's [NAME_n] into one sequence, the same word always getting the same token, and keeps every
    token it handed out, so the final answer is restored from the whole conversation.

    Text from outside (a message, a tool's result) can contain something that LOOKS like a token,
    typed or copied. It is defused first ("[TRAINEE_3]" becomes "TRAINEE_3"), so it cannot be restored
    as a name it never stood for.
    """

    def __init__(self, roster: Roster) -> None:
        self._anonymizer = Anonymizer(roster)
        self._people: dict[str, Person] = {}
        self._unresolved: dict[str, str] = {}  # [NAME_n] -> the word as first written
        self._by_word: dict[str, str] = {}  # the normalized word -> its [NAME_n]

    def anonymize(self, text: str, *, prefer: Iterable[Person] = ()) -> str:
        part = self._anonymizer.anonymize(defuse_tokens(text), prefer=prefer)
        renumber: dict[str, str] = {}
        for token, word in part.unresolved.items():
            key = _normalize(word)
            if key not in self._by_word:
                self._by_word[key] = f"[NAME_{len(self._by_word) + 1}]"
                self._unresolved[self._by_word[key]] = word
            renumber[token] = self._by_word[key]
        self._people.update(part.people)
        return _NAME_TOKEN.sub(lambda m: renumber.get(m.group(0), m.group(0)), part.text)

    def view(self) -> AnonymizedText:
        """Everything handed out so far, as one AnonymizedText: for deanonymize() and for checking that
        a model's answer uses only tokens it was given."""
        return AnonymizedText(
            text="",
            people=MappingProxyType(dict(self._people)),
            unresolved=MappingProxyType(dict(self._unresolved)),
            _key=_CREATED_BY_ANONYMIZER,
        )

    def word_for(self, token: str) -> str | None:
        """The word behind a [NAME_n] token, as the counselor wrote it ("which יעל?" needs the word)."""
        return self._unresolved.get(token)

    def contains_known_name(self, text: str) -> bool:
        return self._anonymizer.contains_known_name(text)


_NAME_TOKEN = re.compile(r"\[NAME_\d+\]")


def defuse_tokens(text: str) -> str:
    """Brackets off anything that looks like a token, so only the anonymizer creates tokens."""
    return _TOKEN.sub(lambda m: m.group(0)[1:-1], text)


def tokens_in(text: str) -> list[str]:
    """Every token in a text, in order. Used to validate a model's answer."""
    return _TOKEN.findall(text)


def load_roster(staff: StaffRepo, trainees: TraineeRepo) -> Roster:
    """Everyone whose name may appear: all staff, and all trainees in every status (a trainee
    who left is still a person whose name must not leave the computer)."""
    people = [Person(PersonKind.STAFF, s.key, *_split_name(s.full_name)) for s in staff.list_all()]
    people += [Person(PersonKind.TRAINEE, t.key, t.first_name, t.last_name) for t in trainees.list_all()]
    return Roster(people)


def _forms_of(person: Person) -> list[str]:
    """Full name, first name, last name, and each word of a first or last name with several
    (split on spaces AND hyphens: "בר-און" gives "בר" and "און" too)."""
    forms = [person.full_name, person.first_name, person.last_name]
    for name in (person.first_name, person.last_name):
        words = _parts(name)
        if len(words) > 1:
            forms += [word for word in words if len(word) > 1]
    return forms


def _split_name(full_name: str) -> tuple[str, str]:
    """Staff have one full_name field: the first word is the first name, the rest the last name."""
    first, _, last = " ".join(full_name.split()).partition(" ")
    return first, last
