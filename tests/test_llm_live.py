"""Against REAL models. Skipped by default and in CI. On your computer:

    uv run pytest --run-llm -m llm            # needs ANTHROPIC_API_KEY in .env, and Ollama running

They check what fakes cannot: that the key works, that the local model is reachable,
and that the connection prompt really produces a short Hebrew answer.
"""

import re

import pytest

from pacer.ai import load_prompt
from pacer.ai.clients import claude_client, ollama_client

pytestmark = pytest.mark.llm
HEBREW = re.compile(r"[֐-׿]")


@pytest.mark.parametrize("make_client", [claude_client, ollama_client], ids=["claude", "ollama"])
def test_model_answers_in_hebrew(make_client):
    response = make_client().complete(load_prompt("connection_check"), "שלום, מה שלומך?", max_tokens=150)

    assert HEBREW.search(response.text), response.text
    assert response.output_tokens > 0
    assert response.prompt_ref == "connection_check@v1"


def test_claude_reports_a_real_cost():
    response = claude_client().complete(load_prompt("connection_check"), "שלום", max_tokens=50)
    assert 0 < response.cost_usd < 0.01  # a greeting costs a tiny fraction of a cent
