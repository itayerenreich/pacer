"""What a call costs, from an explicit price table we maintain.

Why our own table instead of litellm.completion_cost():
* The price table bundled with the pinned litellm version does not know the Claude 5.5
  models, and for an unknown model it returns 0.0: calls that cost money would look free.
* The alternative, letting litellm download its table on every import, is a network call
  we did not choose and prices that change without a commit.

Rules: a model missing from this table has an UNKNOWN cost (None), never 0. Local models
cost 0 per call (that is decided by the client, which knows the model runs locally).
When Anthropic changes prices, update this file, the date, and the tests.
"""

from dataclasses import dataclass

SOURCE = "https://platform.claude.com/docs/en/about-claude/pricing"
CHECKED_ON = "2026-10-14"


@dataclass(frozen=True)
class Price:
    """US dollars per million tokens. Some models charge more for very long prompts."""

    input: float
    output: float
    long_prompt_tokens: int | None = None  # above this many input tokens, the long rates apply
    long_input: float | None = None
    long_output: float | None = None


PRICES: dict[str, Price] = {
    "anthropic/claude-haiku-5-5": Price(
        input=0.10, output=0.50, long_prompt_tokens=100_000, long_input=0.50, long_output=2.50
    ),
    "anthropic/claude-sonnet-5-5": Price(input=2.00, output=10.00),
    "anthropic/claude-opus-5-5": Price(input=4.00, output=20.00),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """The cost of one call, or None when the model's price is not in the table."""
    price = PRICES.get(model)
    if price is None:
        return None
    is_long = price.long_prompt_tokens is not None and input_tokens > price.long_prompt_tokens
    input_rate = price.long_input if is_long else price.input
    output_rate = price.long_output if is_long else price.output
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
