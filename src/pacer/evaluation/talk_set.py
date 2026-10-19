"""Invented personal talks, with what the talk pipeline should make of them.

Each case is a counselor's notes about one trainee of the demo academy, and:
* progress:  goals the talk is about, with the range a sensible new progress falls in. `required`
             goals must get a suggestion in range; the others may get one, in range, or none. A goal
             that is not listed must not get one at all.
* notes:     the notes worth remembering (kind, and the gist in Hebrew) for a judge to match.
* forbidden: words that must never appear in what is proposed (health details, removed before sending).
* mentions:  every word in the notes that is also a name, marked with what it MEANS here: a person
             ("trainee:אגם אוחיון", "staff:נועה לוי") or "word" ("לאגם", to the lake). This is the
             answer key for the open question from stage 3: how often does anonymization hide a word
             as a name ([NAME_n]), or worse, tie a word to a real trainee? (masking.py)

People are named, not numbered: ids depend on the database, a demo trainee's name does not.
Goals are named by their title, which is unique per trainee in the demo academy.
"""

import json
import re
from datetime import date
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pacer.domain import Goal, NoteKind, StaffMember, Trainee
from pacer.repositories.interfaces import GoalRepo, StaffRepo, TraineeRepo

_MEANING = re.compile(r"^(word|trainee:.+|staff:.+)$")


def words_of(text: str) -> list[str]:
    """The text split into words, with punctuation trimmed from their ends: the unit mentions count in."""
    return [w for w in (_trim(w) for w in text.split()) if w]


def _trim(word: str) -> str:
    return word.strip(".,:;!?()\"'״׳-–")  # not [ ]: tokens keep their brackets


class Mention(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    word: str = Field(min_length=2, description="As written, with any prefix letter: 'לאגם'")
    occurrence: int = Field(default=1, ge=1, description="Which time this word appears in the notes")
    means: str = Field(pattern=_MEANING.pattern, description="'word', 'trainee:<full name>' or 'staff:<full name>'")

    @property
    def is_word(self) -> bool:
        return self.means == "word"


class ExpectedProgress(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    goal: str = Field(description="The goal's title")
    low: int = Field(ge=0, le=100)
    high: int = Field(ge=0, le=100)
    required: bool = True

    @model_validator(mode="after")
    def check_range(self) -> Self:
        if self.low > self.high:
            raise ValueError(f"'{self.goal}': low {self.low} is above high {self.high}")
        return self


class ExpectedNote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: NoteKind
    gist: str = Field(min_length=5)


class TalkCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^T\d{2,3}$")
    tests: str = Field(description="What this case is for, in one line")
    trainee: str
    counselor: str
    held_on: date
    notes: str = Field(min_length=20)
    progress: list[ExpectedProgress] = Field(default_factory=list)
    expected_notes: list[ExpectedNote] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    mentions: list[Mention] = Field(default_factory=list)
    reviewed: bool = False

    @model_validator(mode="after")
    def check_mentions_are_in_the_notes(self) -> Self:
        words = words_of(self.notes)
        for m in self.mentions:
            if words.count(m.word) < m.occurrence:
                raise ValueError(f"{self.id}: '{m.word}' does not appear {m.occurrence} time(s) in the notes")
        keys = [(m.word, m.occurrence) for m in self.mentions]
        if len(set(keys)) != len(keys):
            raise ValueError(f"{self.id}: a mention is marked twice")
        goals = [p.goal for p in self.progress]
        if len(set(goals)) != len(goals):
            raise ValueError(f"{self.id}: a goal is listed twice")
        return self


class TalkEvalSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: int = Field(ge=1)
    about: str
    cases: list[TalkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def check_unique_ids(self) -> Self:
        ids = [c.id for c in self.cases]
        if len(set(ids)) != len(ids):
            raise ValueError("Two cases have the same id")
        return self

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"


def load_talk_set(path: Path) -> TalkEvalSet:
    return TalkEvalSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


# --- the case, in the academy ------------------------------------------------------------------------


class ResolvedTalk(BaseModel):
    """A case with its people and goals found in the academy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    case: TalkCase
    trainee: Trainee
    counselor: StaffMember
    goals: list[Goal]  # all of the trainee's goals: the pipeline sends them all

    def goal(self, title: str) -> Goal:
        return next(g for g in self.goals if g.title == title)


class CaseError(Exception):
    """A case does not fit the academy: a name, a goal or a counselor that is not there."""


def resolve(case: TalkCase, staff: StaffRepo, trainees: TraineeRepo, goals: GoalRepo) -> ResolvedTalk:
    problems: list[str] = []
    trainee = next((t for t in trainees.list_all() if t.full_name == case.trainee), None)
    counselor = next((s for s in staff.list_all() if s.full_name == case.counselor), None)
    if trainee is None:
        problems.append(f"no trainee named {case.trainee}")
    if counselor is None:
        problems.append(f"no staff member named {case.counselor}")
    their_goals = goals.list_by_trainee(trainee.key) if trainee is not None else []
    titles = {g.title for g in their_goals}
    problems += [f"{case.trainee} has no goal '{p.goal}'" for p in case.progress if trainee and p.goal not in titles]
    for m in case.mentions:
        kind, _, name = m.means.partition(":")
        people = trainees.list_all() if kind == "trainee" else staff.list_all() if kind == "staff" else []
        if not m.is_word and not any(p.full_name == name for p in people):
            problems.append(f"'{m.word}' means {m.means}, who is not in the academy")
    if problems:
        raise CaseError(f"{case.id}: " + "; ".join(problems))
    assert trainee is not None and counselor is not None
    return ResolvedTalk(case=case, trainee=trainee, counselor=counselor, goals=their_goals)
