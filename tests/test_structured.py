"""Structured output: valid data in, every kind of bad answer handled, nothing guessed."""

import json
import logging

import pytest
from fake_llm import FakeLLMClient
from pydantic import SecretStr

from pacer.ai import LiteLLMClient, LLMResponse, load_prompt
from pacer.ai import llm_client as llm_module
from pacer.ai.checks import SAMPLE_TEXT, TextDigest
from pacer.ai.structured import (
    MAX_ATTEMPTS,
    StructuredOutputError,
    StructuredResult,
    complete_structured,
)

PROMPT = load_prompt("structured_check")
GOOD = {
    "language": "he",
    "summary": "המחזור יצא להתנדבות בצפון ושוחח על אחריות.",
    "topics": ["התנדבות", "אחריות"],
}
GOOD_JSON = json.dumps(GOOD, ensure_ascii=False)


def run(*answers: str, **fake_options) -> tuple[StructuredResult[TextDigest], FakeLLMClient]:
    fake = FakeLLMClient(answers=list(answers), **fake_options)
    return complete_structured(fake, PROMPT, SAMPLE_TEXT, TextDigest), fake


# --- the happy path ---------------------------------------------------------------------


def test_valid_json_becomes_a_validated_object():
    result, fake = run(GOOD_JSON)

    assert isinstance(result.value, TextDigest)
    assert result.value.topics == ["התנדבות", "אחריות"]
    assert result.attempts == 1
    assert result.prompt_ref == "structured_check@v1"


def test_the_schema_goes_to_the_provider_and_into_the_message():
    _, fake = run(GOOD_JSON)
    [call] = fake.calls
    assert call.response_schema == TextDigest.model_json_schema()
    assert call.user_message.startswith(SAMPLE_TEXT)  # the real input first
    assert '"topics"' in call.user_message  # then the schema, for models without native support


@pytest.mark.parametrize(
    "wrapped",
    [
        f"```json\n{GOOD_JSON}\n```",
        f"```\n{GOOD_JSON}\n```",
        f"הנה התשובה:\n{GOOD_JSON}",
        f"{GOOD_JSON}\nבהצלחה!",
    ],
    ids=["json-fence", "plain-fence", "text-before", "text-after"],
)
def test_common_wrappings_are_tolerated(wrapped):
    result, _ = run(wrapped)
    assert result.value.language == "he"


# --- a bad answer gets one correction --------------------------------------------------------


def test_invalid_answer_is_corrected_with_the_exact_errors():
    too_short = json.dumps({**GOOD, "summary": "קצר"}, ensure_ascii=False)
    result, fake = run(too_short, GOOD_JSON)

    assert result.attempts == 2
    first, second = fake.calls
    assert "rejected" not in first.user_message
    assert "summary: String should have at least 10 characters" in second.user_message
    assert "קצר" not in second.user_message.split("rejected")[1]  # errors describe, never quote input


def test_not_json_at_all_is_corrected_too():
    result, fake = run("אני לא בטוח מה לענות", GOOD_JSON)
    assert result.attempts == 2
    assert "Invalid JSON" in fake.calls[1].user_message


def test_unknown_fields_are_rejected():
    """A schema with extra='forbid' does not let the model add fields we did not ask for."""
    extra = json.dumps({**GOOD, "trainee_name": "יעל"}, ensure_ascii=False)
    result, fake = run(extra, GOOD_JSON)
    assert result.attempts == 2
    assert "trainee_name" in fake.calls[1].user_message


def test_rules_the_provider_cannot_enforce_are_still_enforced():
    """Claude's native mode drops maxItems; Pydantic still enforces 'at most 3 topics'."""
    four_topics = json.dumps({**GOOD, "topics": ["א", "ב", "ג", "ד"]}, ensure_ascii=False)
    with pytest.raises(StructuredOutputError):
        run(four_topics, four_topics)


# --- and then a clear failure, never a guess -------------------------------------------------


def test_two_bad_answers_raise_with_the_reasons():
    with pytest.raises(StructuredOutputError) as caught:
        run('{"language": "en"}')

    error = caught.value
    assert error.attempts == MAX_ATTEMPTS
    assert any(reason.startswith("language:") for reason in error.reasons)
    assert any(reason.startswith("summary:") for reason in error.reasons)


def test_a_truncated_answer_fails_at_once_without_retrying():
    """Asking again with the same max_tokens would be cut off again: fail fast, fix the route."""
    with pytest.raises(StructuredOutputError, match="cut off") as caught:
        run(GOOD_JSON[:30], truncated=True)
    assert caught.value.attempts == 1


def test_the_answer_text_never_reaches_errors_or_logs(caplog):
    caplog.set_level(logging.INFO)
    secret = "יעל מזרחי"
    with pytest.raises(StructuredOutputError) as caught:
        run(json.dumps({"language": "he", "summary": secret, "topics": []}, ensure_ascii=False))
    assert secret not in str(caught.value)
    assert secret not in " ".join(caught.value.reasons)
    assert secret not in caplog.text
    assert "Structured output rejected" in caplog.text


# --- cost ------------------------------------------------------------------------------------


def response(cost: float | None) -> LLMResponse:
    return LLMResponse(
        text="{}",
        model="m",
        prompt_ref="p@v1",
        input_tokens=1,
        output_tokens=1,
        cost_usd=cost,
        latency_ms=1,
        truncated=False,
    )


def test_cost_adds_up_all_attempts():
    result = StructuredResult(value=TextDigest(**GOOD), responses=[response(0.001), response(0.002)])
    assert result.cost_usd == pytest.approx(0.003)


def test_cost_is_unknown_if_any_attempt_is_unknown():
    result = StructuredResult(value=TextDigest(**GOOD), responses=[response(0.001), response(None)])
    assert result.cost_usd is None


# --- through the real LiteLLM client (litellm's mock, no network) ----------------------------


@pytest.fixture
def sent(monkeypatch):
    requests: list[dict] = []
    real = llm_module.litellm.completion

    def capture(**request):
        requests.append(request)
        return real(mock_response=GOOD_JSON, **request)

    monkeypatch.setattr(llm_module.litellm, "completion", capture)
    return requests


def test_claude_gets_the_schema_as_a_native_response_format(sent):
    client = LiteLLMClient(
        "anthropic/claude-haiku-5-5", runs_locally=False, api_key=SecretStr("sk-test"), retries=0
    )
    result = complete_structured(client, PROMPT, SAMPLE_TEXT, TextDigest)

    assert result.value.language == "he"
    assert sent[0]["response_format"]["json_schema"]["schema"] == TextDigest.model_json_schema()


def test_a_model_without_native_support_gets_only_the_instructions(sent):
    client = LiteLLMClient("ollama/llama3.2", runs_locally=True, api_base="http://localhost:11434", retries=0)
    result = complete_structured(client, PROMPT, SAMPLE_TEXT, TextDigest)

    assert result.value.language == "he"
    assert "response_format" not in sent[0]
