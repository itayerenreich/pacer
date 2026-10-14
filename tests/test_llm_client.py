"""LiteLLMClient, without network: litellm's mock_response returns a fake answer (or raises a
fake provider error) after running all of litellm's real request handling.
"""

import logging
import os
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from pacer.ai import (
    LiteLLMClient,
    LLMAuthError,
    LLMConfigError,
    LLMError,
    LLMRateLimitError,
    LLMUnavailableError,
    load_prompt,
)
from pacer.ai import llm_client as module
from pacer.ai.clients import claude_client, ollama_client
from pacer.config import get_ai_settings

PROMPT = load_prompt("connection_check")
SECRET_MESSAGE = "יעל מזרחי עזבה את המכינה"  # must never appear in logs


@pytest.fixture
def sent(monkeypatch):
    """Capture the exact request litellm receives, and answer with litellm's mock."""
    requests: list[dict] = []
    real_completion = module.litellm.completion

    def capture(**request):
        requests.append(request)
        return real_completion(mock_response=capture.answer, **request)

    capture.answer = "שלום! זו בדיקת חיבור."
    monkeypatch.setattr(module.litellm, "completion", capture)
    capture.requests = requests
    return capture


def cloud(**kwargs) -> LiteLLMClient:
    return LiteLLMClient(
        "anthropic/claude-haiku-5-5", runs_locally=False, api_key=SecretStr("sk-test"), retries=0, **kwargs
    )


def local(**kwargs) -> LiteLLMClient:
    return LiteLLMClient(
        "ollama/llama3.2", runs_locally=True, api_base="http://localhost:11434", retries=0, **kwargs
    )


# --- a successful call ------------------------------------------------------------------


def test_answer_and_its_record(sent):
    response = cloud().complete(PROMPT, "שלום", max_tokens=100)

    assert response.text == "שלום! זו בדיקת חיבור."
    assert response.prompt_ref == "connection_check@v1"
    assert response.model == "anthropic/claude-haiku-5-5"
    assert response.input_tokens > 0 and response.output_tokens > 0
    # litellm's mock reports 10 input and 20 output tokens; Haiku 5.5: $0.10 / $0.50 per million
    assert response.cost_usd == pytest.approx((10 * 0.10 + 20 * 0.50) / 1_000_000)
    assert response.truncated is False


def test_the_system_prompt_is_always_sent_first(sent):
    cloud().complete(PROMPT, "שלום")
    [request] = sent.requests
    assert request["messages"] == [
        {"role": "system", "content": PROMPT.system},
        {"role": "user", "content": "שלום"},
    ]


def test_a_model_without_a_known_price_has_unknown_cost_not_zero(sent):
    client = LiteLLMClient(
        "anthropic/claude-future-9", runs_locally=False, api_key=SecretStr("sk-test"), retries=0
    )
    assert client.complete(PROMPT, "שלום").cost_usd is None


def test_a_local_model_costs_nothing(sent):
    assert local().complete(PROMPT, "שלום").cost_usd == 0.0


def test_temperature_is_sent_only_when_set(sent):
    """Claude 5.5 rejects temperature, so it is never sent unless a model is configured with it."""
    cloud().complete(PROMPT, "שלום")
    local(temperature=0.2).complete(PROMPT, "שלום")
    assert "temperature" not in sent.requests[0]
    assert sent.requests[1]["temperature"] == 0.2


def test_the_key_goes_with_the_request_not_into_the_environment(sent):
    cloud().complete(PROMPT, "שלום")
    assert sent.requests[0]["api_key"] == "sk-test"
    assert "ANTHROPIC_API_KEY" not in os.environ or os.environ["ANTHROPIC_API_KEY"] != "sk-test"


def test_empty_message_is_rejected_before_any_call(sent):
    with pytest.raises(ValueError):
        cloud().complete(PROMPT, "   ")
    assert sent.requests == []


def test_a_cut_off_answer_is_marked_truncated():
    raw = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="חצי תשוב"), finish_reason="length")],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=5),
    )
    response = local()._to_response(raw, PROMPT, started=0.0)
    assert response.truncated is True


# --- privacy: what reaches the logs ------------------------------------------------------


def test_logs_have_the_prompt_version_but_never_the_text(sent, caplog):
    caplog.set_level(logging.INFO, logger="pacer.ai.llm_client")
    sent.answer = "תשובה עם שם: יעל"
    cloud().complete(PROMPT, SECRET_MESSAGE)

    assert "prompt=connection_check@v1" in caplog.text
    assert "cost=$" in caplog.text
    assert SECRET_MESSAGE not in caplog.text
    assert "יעל" not in caplog.text


def test_no_price_table_download_on_import():
    assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "True"


# --- errors become our own -----------------------------------------------------------------


def provider_error(error_class):
    return error_class(message="provider says no", llm_provider="anthropic", model="claude-haiku-5-5")


@pytest.mark.parametrize(
    ("raised", "expected"),
    [
        (module.litellm_errors.AuthenticationError, LLMAuthError),
        (module.litellm_errors.RateLimitError, LLMRateLimitError),
        (module.litellm_errors.ServiceUnavailableError, LLMUnavailableError),
    ],
)
def test_provider_errors_become_pacer_errors(sent, raised, expected):
    sent.answer = provider_error(raised)
    with pytest.raises(expected) as caught:
        cloud().complete(PROMPT, "שלום")
    assert isinstance(caught.value, LLMError)  # one base class for callers to catch


def test_ollama_not_running_is_unavailable():
    """A real connection attempt to a port where nothing listens."""
    client = LiteLLMClient(
        "ollama/llama3.2", runs_locally=True, api_base="http://127.0.0.1:9", retries=0, timeout_s=5
    )
    with pytest.raises(LLMUnavailableError):
        client.complete(PROMPT, "שלום")


# --- building clients ---------------------------------------------------------------------


def test_claude_without_a_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("pacer.config.AISettings.model_config", {"env_file": None, "extra": "ignore"})
    get_ai_settings.cache_clear()
    try:
        with pytest.raises(LLMConfigError, match="ANTHROPIC_API_KEY is missing"):
            claude_client()
    finally:
        get_ai_settings.cache_clear()


def test_clients_declare_where_the_data_goes(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    get_ai_settings.cache_clear()
    try:
        assert claude_client().runs_locally is False
        assert ollama_client().runs_locally is True
    finally:
        get_ai_settings.cache_clear()
