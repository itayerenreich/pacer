"""Risk situations the safety layer must find, and everyday sentences it must leave alone.

No model is involved (ai/safety.py is code), so this set runs in the regular test suite, on every pull
request, not in the paid evaluation runner:

* cases:        a risk; must be found, with at least the expected kinds. The bar is ALL of them: a
                missed risk means a counselor gets advice where they needed the procedure.
* near_misses:  a risk word in its everyday academy sense; must not become a referral. Each one is the
                reason a word is kept off the list ("נשק", "רצח", "מטריד"...), so a later "fix" that
                adds the word back fails a test that explains why.

A new phrasing that slipped through goes into a case first (and fails), then into the patterns.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pacer.ai.safety import RiskKind, check_safety


class SafetyCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[SH]\d{2,}$")
    text: str = Field(min_length=5)
    expect: list[RiskKind] = Field(min_length=1)


class NearMiss(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[NE]\d{2,}$")
    text: str = Field(min_length=5)
    why: str = Field(min_length=3)


class SafetyEvalSet(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    version: int = Field(ge=1)
    about: str
    cases: list[SafetyCase] = Field(min_length=1)
    near_misses: list[NearMiss] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_unique(self) -> Self:
        ids = [c.id for c in self.cases] + [n.id for n in self.near_misses]
        texts = [c.text for c in self.cases] + [n.text for n in self.near_misses]
        for what, values in (("id", ids), ("text", texts)):
            if duplicates := sorted({v for v in values if values.count(v) > 1}):
                raise ValueError(f"Two cases have the same {what}: {duplicates}")
        return self

    @property
    def ref(self) -> str:
        return f"{self.name}@v{self.version}"


def load_safety_set(path: Path) -> SafetyEvalSet:
    return SafetyEvalSet.model_validate(json.loads(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class SafetyResult:
    found: int  # risk cases found with every expected kind
    total: int
    missed: list[str]  # "S07: expected self_harm, found nothing"
    false_alarms: list[str]  # "N04: violence"
    near_misses: int

    @property
    def recall(self) -> float:
        return self.found / self.total


def evaluate(eval_set: SafetyEvalSet) -> SafetyResult:
    missed = []
    for case in eval_set.cases:
        kinds = check_safety(case.text).kinds
        if absent := [k for k in case.expect if k not in kinds]:
            got = ", ".join(sorted(kinds)) or "nothing"
            missed.append(f"{case.id}: expected {', '.join(absent)}, found {got}")
    false_alarms = [
        f"{n.id}: {', '.join(sorted(kinds))}"
        for n in eval_set.near_misses
        if (kinds := check_safety(n.text).kinds)
    ]
    return SafetyResult(
        found=len(eval_set.cases) - len(missed),
        total=len(eval_set.cases),
        missed=missed,
        false_alarms=false_alarms,
        near_misses=len(eval_set.near_misses),
    )
