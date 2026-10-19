"""Questions for the library, with what a right answer looks like.

Each case says:
* sources:   EVERY source that answers it. A question can have more than one right source (the first
             run taught us that: two "misses" were right answers we had not listed).
* coverage:  how well the library can answer it, as the answerer reports it (ai.answer.Coverage).
             A LIST of acceptable values, because the line between "direct" and "partial" is a
             judgment: a source about teachers of children, asked about counselors, is fairly either.
             "none" is never mixed with the others: either the library has a source, or it does not.
* kind:      what the case tests, so results can be read per kind.
* judge:     what a judge (a person, or a model in task 8) must check beyond the sources, e.g.
             "never suggests hitting".
* reviewed:  whether Itay checked it. Written by Claude from the texts; reported separately until
             reviewed, so an unchecked label never hides inside a good-looking number.
"""

import json
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pacer.ai.answer import Coverage


class CaseKind(StrEnum):
    APPLIED = "applied"  # a counselor's situation, answered by applying a source
    ABOUT_SOURCE = "about_source"  # asks what a source itself says
    HARD_NEGATIVE = "hard_negative"  # educational, close to the library, but no source answers it
    OFF_TOPIC = "off_topic"  # nothing to do with the library
    SAFETY = "safety"  # a risk to someone: must go to the manager and the procedure, whatever is found


class LibraryCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^L\d{2,3}$")
    question: str = Field(min_length=5, max_length=1000)
    kind: CaseKind
    sources: list[str] = Field(description="Titles of every source that answers it; empty: none does")
    coverage: list[Coverage] = Field(min_length=1)
    judge: list[str] = Field(default_factory=list, description="What a right answer must (not) do")
    reviewed: bool = False

    @model_validator(mode="after")
    def check_consistent(self) -> Self:
        has_sources = bool(self.sources)
        if has_sources and Coverage.NONE in self.coverage:
            raise ValueError(f"{self.id}: lists sources, so its coverage cannot be 'none'")
        if not has_sources and self.coverage != [Coverage.NONE]:
            raise ValueError(f"{self.id}: no source answers it, so its coverage must be exactly ['none']")
        if self.kind in (CaseKind.HARD_NEGATIVE, CaseKind.OFF_TOPIC) and has_sources:
            raise ValueError(f"{self.id}: a {self.kind} case has no source, by definition")
        if self.kind in (CaseKind.APPLIED, CaseKind.ABOUT_SOURCE) and not has_sources:
            raise ValueError(f"{self.id}: a {self.kind} case is answered by some source; list it")
        if len(set(self.sources)) != len(self.sources):
            raise ValueError(f"{self.id}: a source is listed twice")
        return self

    @property
    def has_source(self) -> bool:
        return bool(self.sources)


class LibraryEvalSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: int = Field(ge=1)
    about: str
    cases: list[LibraryCase] = Field(min_length=1)

    @model_validator(mode="after")
    def check_unique(self) -> Self:
        for field_name in ("id", "question"):
            values = [getattr(c, field_name) for c in self.cases]
            if duplicates := sorted({v for v in values if values.count(v) > 1}):
                raise ValueError(f"Two cases have the same {field_name}: {duplicates}")
        return self

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"


def load_library_set(path: Path) -> LibraryEvalSet:
    return LibraryEvalSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


def unknown_sources(eval_set: LibraryEvalSet, titles: set[str]) -> list[str]:
    """Expected titles that are not in the library: a typo in a set would silently count as a miss."""
    return sorted({s for c in eval_set.cases for s in c.sources} - titles)
