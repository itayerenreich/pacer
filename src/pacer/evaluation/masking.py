"""What anonymization did to every word that is also a name: measured against the answer key.

No model and no network: the exact message the talk pipeline would send is built
(prepare_talk_message, the pipeline's own function), and each marked word is looked up in it.

    a PERSON           right token     the model knows who it is                     right
                       [NAME_n]        hidden, but the model cannot tell who         safe, information lost
                       another person  tied to the wrong trainee                     safe, but wrong
                       the name itself a name left the computer                      privacy failure
    a WORD             unchanged       the model reads the word                      right
                       [NAME_n]        over-masked: an odd sentence for the model    safe, the cost we chose
                       a person        a word tied to a real trainee                 wrong, and misleading

Finding a marked word in the sent text: the notes are compared word by word with their anonymized
form (difflib). A full name is two words that became one token, so blocks can differ in length.
"""

import difflib
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from pacer.ai.anonymization import Roster
from pacer.ai.sensitive import remove_sensitive_sentences
from pacer.ai.talk_pipeline import prepare_talk_message

from .talk_set import Mention, ResolvedTalk, words_of

_NOTES_START = "The counselor's notes:\n"
_PERSON_TOKEN = re.compile(r"\[(TRAINEE|STAFF)_(\d+)\]")
_NEUTRAL_TOKEN = re.compile(r"\[NAME_\d+\]")


class Outcome(StrEnum):
    PERSON_RIGHT = "person_right"
    PERSON_NEUTRAL = "person_neutral"
    PERSON_WRONG = "person_wrong"
    PERSON_LEAKED = "person_leaked"
    WORD_KEPT = "word_kept"
    WORD_NEUTRAL = "word_neutral"
    WORD_AS_PERSON = "word_as_person"


SAFE_FOR_PRIVACY = frozenset(set(Outcome) - {Outcome.PERSON_LEAKED})


@dataclass(frozen=True)
class MentionResult:
    case_id: str
    mention: Mention
    sent_as: str  # the word as the model receives it
    outcome: Outcome


def _token_for(means: str, roster: Roster) -> str:
    kind, _, name = means.partition(":")
    person = next(p for p in roster.people if p.kind.value.lower() == kind and p.full_name == name)
    return person.token


def _sent_words(talk: ResolvedTalk, roster: Roster) -> tuple[list[str], list[str]]:
    """The notes as kept (before anonymization) and as sent, both as word lists."""
    case = talk.case
    message, _ = prepare_talk_message(roster, talk.trainee, talk.counselor, case.held_on, talk.goals, case.notes)
    kept, _ = remove_sensitive_sentences(case.notes)
    sent_notes = message.text.split(_NOTES_START, 1)[1]
    return words_of(kept), words_of(sent_notes)


def _aligned(before: list[str], after: list[str]) -> dict[int, str]:
    """For each word index of `before`, the word it became in `after`."""
    mapping: dict[int, str] = {}
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            mapping.update({i1 + k: after[j1 + k] for k in range(i2 - i1)})
        elif op == "replace":
            for k in range(i2 - i1):  # 2 words -> 1 token (a full name): both map to the token
                mapping[i1 + k] = after[min(j1 + k, j2 - 1)]
    return mapping


def _index_of(words: list[str], mention: Mention) -> int | None:
    seen = 0
    for i, w in enumerate(words):
        if w == mention.word:
            seen += 1
            if seen == mention.occurrence:
                return i
    return None


def classify(mention: Mention, sent_as: str, expected_token: str | None) -> Outcome:
    person = _PERSON_TOKEN.search(sent_as)
    neutral = _NEUTRAL_TOKEN.search(sent_as)
    if mention.is_word:
        if person:
            return Outcome.WORD_AS_PERSON
        return Outcome.WORD_NEUTRAL if neutral else Outcome.WORD_KEPT
    if person:
        return Outcome.PERSON_RIGHT if person.group(0) == expected_token else Outcome.PERSON_WRONG
    return Outcome.PERSON_NEUTRAL if neutral else Outcome.PERSON_LEAKED


def measure(talk: ResolvedTalk, roster: Roster) -> list[MentionResult]:
    kept, sent = _sent_words(talk, roster)
    mapping = _aligned(kept, sent)
    full = words_of(talk.case.notes)
    results = []
    for m in talk.case.mentions:
        if full.count(m.word) != kept.count(m.word):
            # Counting occurrences across a removed sentence would mark the wrong word: keep marked
            # words out of health sentences (they are removed before sending; nothing to measure there).
            raise ValueError(f"{talk.case.id}: '{m.word}' is also in a sentence removed before sending")
        index = _index_of(kept, m)
        if index is None or index not in mapping:
            raise ValueError(f"{talk.case.id}: could not find '{m.word}' #{m.occurrence} in what is sent")
        expected = None if m.is_word else _token_for(m.means, roster)
        sent_as = mapping[index]
        results.append(MentionResult(talk.case.id, m, sent_as, classify(m, sent_as, expected)))
    return results


def summary(results: list[MentionResult]) -> Counter[Outcome]:
    return Counter(r.outcome for r in results)
