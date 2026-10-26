"""The safety gate: the phrase list first, then a model as a second net that can only ADD a referral.

    gate.assess(question, anonymizer)  ->  SafetyDecision(referral or None, found_by)

    1. phrases   check_safety() on the text as written, in code, on this computer. A match is final:
                 the referral is returned and no model is asked at all.
    2. model     otherwise, Claude reads the text and answers one question: is this a risk situation?
                 Names are anonymized and health WORDS masked first (not whole sentences, as for the
                 coach: a risk scene inside a sentence about health must still be screened).

Why two layers (decision 2026-10-25, after measuring): the phrase list found 30 of 60 risk sentences
written by someone who never saw it (evals/safety-heldout-v1.json). Phrases find WORDS; subtle risk is
described as a SCENE ("stood on the roof at night", "came back with a black eye and asked us not to call
his parents"). A model reads scenes.

Privacy: the screen sees the anonymized question, as the coach would a moment later. One difference,
on purpose: where the coach drops a whole sentence that mentions health, the screen gets that sentence
with only its health words masked, so a scene in it is still screened. No medical word leaves.

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
from .sensitive import HEALTH_MASK, mask_sensitive_words
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
    kinds: frozenset[RiskKind] = frozenset()  # internal only: for the logs and the evaluation, never shown
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
            return SafetyDecision(referral_for(found), FoundBy.PHRASES, found.kinds)

        # Health WORDS are masked, not whole sentences removed (unlike the coach): a risk scene in a
        # sentence that mentions a psychologist must still be screened (independent review, 2026-10-25).
        # The medical words never leave the computer; the scene does, anonymized.
        minimized, _masked = mask_sensitive_words(text)
        if not any(ch.isalpha() for ch in minimized.replace(HEALTH_MASK, "")):
            return SafetyDecision(None)  # nothing but health words: nothing to screen, nothing sent
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
        kinds = frozenset(answer.kinds)
        return SafetyDecision(
            referral_for(SafetyCheck(kinds)), FoundBy.MODEL, kinds, reason=answer.reason, cost_usd=result.cost_usd
        )
