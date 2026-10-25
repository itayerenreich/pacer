"""The safety gate: the phrase list first, then a model as a second net that can only ADD a referral.

    gate.assess(question, anonymizer)  ->  SafetyDecision(referral or None, found_by)

    1. phrases   check_safety() on the text as written, in code, on this computer. A match is final:
                 the referral is returned and no model is asked at all.
    2. model     otherwise, Claude reads the text and answers one question: is this a risk situation?
                 Health sentences are removed and names anonymized first, exactly as for the coach.

Why two layers (decision 2026-10-25, after measuring): the phrase list found 30 of 60 risk sentences
written by someone who never saw it (evals/safety-heldout-v1.json). Phrases find WORDS; subtle risk is
described as a SCENE ("stood on the roof at night", "came back with a black eye and asked us not to call
his parents"). A model reads scenes.

Why this costs no privacy: if the phrases find nothing, the coach sends the same anonymized text to the
same model a moment later anyway. The screen sees nothing the coach would not.

The model can only add: it runs only when the phrases found nothing, and a "no" from it changes nothing.
And it fails CLOSED: if the model is down or its answer is invalid, assess() raises, so the coach gives
no advice that was not screened (the API turns it into 503, "try again"). Never a silent pass.
"""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .anonymization import AnonymizationError, Anonymizer, Person
from .prompts import Prompt, load_prompt
from .routing import AITask, Router
from .safety import Referral, RiskKind, SafetyCheck, check_safety, referral_for
from .sensitive import remove_sensitive_sentences
from .structured import complete_structured

log = logging.getLogger(__name__)

PROMPT_VERSION = 1  # pinned: a used version is never edited, a change is a new version


class ScreenAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk: bool
    kinds: list[RiskKind] = Field(max_length=4, description="The kinds found; empty if and only if risk is false")
    reason: str = Field(max_length=300, description="One short English sentence naming the signal, no names")

    @model_validator(mode="after")
    def kinds_match_risk(self) -> Self:
        if self.risk != bool(self.kinds):
            raise ValueError("kinds must be empty exactly when risk is false")
        return self


class FoundBy(StrEnum):
    PHRASES = "phrases"
    MODEL = "model"


@dataclass(frozen=True)
class SafetyDecision:
    referral: Referral | None
    found_by: FoundBy | None = None
    reason: str | None = None  # the model's one sentence (for the evaluation report; never logged)
    cost_usd: float | None = 0.0

    @property
    def is_risk(self) -> bool:
        return self.referral is not None


class SafetyGate:
    def __init__(self, router: Router, *, prompt: Prompt | None = None) -> None:
        self._router = router
        self._prompt = prompt or load_prompt("risk_screen", PROMPT_VERSION)

    def assess(self, text: str, anonymizer: Anonymizer, *, prefer: tuple[Person, ...] = ()) -> SafetyDecision:
        """prefer: who the text is about (the coach knows the trainee), so an ambiguous first name is them."""
        found = check_safety(text)
        if found.is_risk:
            log.warning("Safety: referral by phrases, kinds=%s", ",".join(sorted(found.kinds)))
            return SafetyDecision(referral_for(found), FoundBy.PHRASES)

        # The same preparation the coach makes before sending: no health sentences, no names. A sentence
        # with a health word never leaves the computer; if it carried a risk the phrases missed, the
        # model cannot see it either. That gap is why every coach answer also carries the procedure line.
        minimized, _removed = remove_sensitive_sentences(text)
        if not minimized.strip():
            return SafetyDecision(None)  # nothing left to send; the coach will not send it either
        safe = anonymizer.anonymize(minimized, prefer=prefer)
        if anonymizer.contains_known_name(safe.text):  # defense in depth
            raise AnonymizationError("A known name is in the text; nothing was sent")

        client = self._router.client_for(AITask.SCREEN_RISK)
        max_tokens = self._router.route(AITask.SCREEN_RISK).max_tokens
        result = complete_structured(client, self._prompt, safe.text, ScreenAnswer, max_tokens=max_tokens)
        answer = result.value
        if not answer.risk:
            return SafetyDecision(None, reason=answer.reason, cost_usd=result.cost_usd)
        log.warning("Safety: referral by the model, kinds=%s", ",".join(sorted(answer.kinds)))
        referral = referral_for(SafetyCheck(frozenset(answer.kinds)))
        return SafetyDecision(referral, FoundBy.MODEL, reason=answer.reason, cost_usd=result.cost_usd)
