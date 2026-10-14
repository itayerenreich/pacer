"""The price table: official prices, the long-prompt tier, and unknown models."""

import pytest

from pacer.ai.pricing import cost_usd


def test_sonnet_one_million_tokens_each_way():
    assert cost_usd("anthropic/claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(2.00 + 10.00)


def test_haiku_normal_prompt():
    assert cost_usd("anthropic/claude-haiku-5-5", 2_000, 500) == pytest.approx(
        (2_000 * 0.10 + 500 * 0.50) / 1e6
    )


def test_haiku_prompt_over_100k_tokens_uses_the_long_rates():
    assert cost_usd("anthropic/claude-haiku-5-5", 100_001, 1_000) == pytest.approx(
        (100_001 * 0.50 + 1_000 * 2.50) / 1e6
    )
    assert cost_usd("anthropic/claude-haiku-5-5", 100_000, 1_000) == pytest.approx(
        (100_000 * 0.10 + 1_000 * 0.50) / 1e6
    )  # exactly 100k is still the normal rate


def test_unknown_model_is_none_never_zero():
    assert cost_usd("anthropic/some-new-model", 1_000, 1_000) is None
