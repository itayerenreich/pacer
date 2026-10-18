"""Structured output: the model returns DATA that matches a Pydantic model, not free text.

    result = complete_structured(client, prompt, transcript, MeetingExtraction)
    result.value          # a validated MeetingExtraction object
    result.attempts       # 1, or 2 if the first answer had to be corrected
    result.cost_usd       # all attempts together

How a call goes:
1. The JSON Schema of the Pydantic model goes to the provider (Claude constrains its answer to
   it). A model WITHOUT native support (llama3.2) gets the schema inside the message instead.
   Not both: for Claude that sent the schema twice and doubled the input tokens of a call.
2. The answer is parsed and validated against the FULL Pydantic model. This is the guarantee:
   the provider cannot enforce every rule (Claude rejects minLength, maximum...), Pydantic can.
3. If validation fails, ONE more attempt: the model gets the exact errors and corrects itself.
4. Two failures, or an answer cut off at max_tokens: StructuredOutputError. Never a guess.

This validates SHAPE (fields, types, lengths). Whether the CONTENT is true (does this staff
member exist, did they attend the meeting?) is checked against the database by the pipeline.
"""

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from .llm_client import LLMClient, LLMError, LLMResponse
from .prompts import Prompt

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2

FORMAT_INSTRUCTIONS = """

---
Answer with ONE JSON object that matches this JSON Schema exactly.
No text before or after the JSON, no Markdown fences.

{schema}"""

NATIVE_FORMAT_INSTRUCTIONS = """

---
Answer with ONE JSON object that matches the response schema you were given."""

CORRECTION = """

---
Your previous answer was rejected for these reasons:
{errors}
Answer again with ONE corrected JSON object that matches the schema."""


class StructuredOutputError(LLMError):
    """The model did not produce a valid object. Carries the reasons, never the answer text
    (it may contain personal data, and errors end up in logs)."""

    def __init__(self, message: str, *, attempts: int, reasons: list[str]) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.reasons = reasons


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    value: T
    responses: list[LLMResponse] = field(repr=False)

    @property
    def attempts(self) -> int:
        return len(self.responses)

    @property
    def cost_usd(self) -> float | None:
        total = 0.0
        for response in self.responses:
            if response.cost_usd is None:
                return None  # one unknown cost makes the total unknown, never an undercount
            total += response.cost_usd
        return total

    @property
    def prompt_ref(self) -> str:
        return self.responses[0].prompt_ref


def complete_structured[T: BaseModel](
    client: LLMClient,
    prompt: Prompt,
    user_message: str,
    schema: type[T],
    *,
    max_tokens: int = 2048,
) -> StructuredResult[T]:
    json_schema = schema.model_json_schema()
    if client.native_schema:
        request = user_message + NATIVE_FORMAT_INSTRUCTIONS
    else:
        request = user_message + FORMAT_INSTRUCTIONS.format(schema=json.dumps(json_schema, ensure_ascii=False))
    responses: list[LLMResponse] = []
    reasons: list[str] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        message = (
            request
            if not reasons
            else request + CORRECTION.format(errors="\n".join(f"- {r}" for r in reasons))
        )
        response = client.complete(prompt, message, max_tokens=max_tokens, response_schema=json_schema)
        responses.append(response)

        if response.truncated:  # half a JSON object cannot be fixed by asking again with the same limit
            raise StructuredOutputError(
                f"{prompt.ref}: answer cut off at max_tokens={max_tokens}; raise it in the route",
                attempts=attempt,
                reasons=["truncated"],
            )
        try:
            value = schema.model_validate_json(_extract_json(response.text))
        except ValidationError as error:
            reasons = _describe(error)
            log.warning(
                "Structured output rejected: prompt=%s schema=%s attempt=%s errors=%s",
                prompt.ref,
                schema.__name__,
                attempt,
                len(reasons),
            )
            continue

        if attempt > 1:
            log.info("Structured output corrected on attempt %s: prompt=%s", attempt, prompt.ref)
        return StructuredResult(value=value, responses=responses)

    raise StructuredOutputError(
        f"{prompt.ref}: no valid {schema.__name__} after {MAX_ATTEMPTS} attempts",
        attempts=MAX_ATTEMPTS,
        reasons=reasons,
    )


def _extract_json(text: str) -> str:
    """From the first '{' to the last '}'. Covers ```json fences and a sentence before or after."""
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def _describe(error: ValidationError) -> list[str]:
    """Pydantic errors as short lines for the model: where and what, never the input values."""
    lines = []
    for item in error.errors(include_url=False, include_input=False, include_context=False):
        where = ".".join(str(part) for part in item["loc"]) or "(the whole answer)"
        lines.append(f"{where}: {item['msg']}")
    return lines
