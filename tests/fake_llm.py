"""FakeLLMClient: a stand-in for a real model in tests. No network, no cost, same answer every time.

It satisfies the LLMClient Protocol, so any code that accepts an LLMClient accepts it.
It also records every call, so a test can check WHAT was sent: which prompt version,
and that no personal data reached a model that is not local.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pacer.ai import LLMResponse, Prompt


@dataclass
class Call:
    prompt: Prompt
    user_message: str
    max_tokens: int
    response_schema: dict[str, Any] | None = None


@dataclass
class FakeLLMClient:
    """answers: a list returned in order (the last one repeats), or a function of the user message."""

    answers: list[str] | Callable[[str], str] = field(default_factory=lambda: ["תשובה לדוגמה"])
    model: str = "fake/model"
    runs_locally: bool = True
    calls: list[Call] = field(default_factory=list)

    truncated: bool = False  # set True to simulate an answer cut off at max_tokens

    def complete(
        self,
        prompt: Prompt,
        user_message: str,
        *,
        max_tokens: int = 1024,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not isinstance(prompt, Prompt):  # same contract as the real client: a versioned prompt
            raise TypeError("complete() needs a Prompt loaded from a prompt file")
        self.calls.append(Call(prompt, user_message, max_tokens, response_schema))
        if callable(self.answers):
            text = self.answers(user_message)
        else:
            text = self.answers[min(len(self.calls), len(self.answers)) - 1]
        return LLMResponse(
            text=text,
            model=self.model,
            prompt_ref=prompt.ref,
            input_tokens=len(prompt.system.split()) + len(user_message.split()),
            output_tokens=len(text.split()),
            cost_usd=0.0,
            latency_ms=0,
            truncated=self.truncated,
        )
