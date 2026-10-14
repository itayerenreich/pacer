"""The fake model used by every future AI test: it must behave like the real interface."""

import pytest
from fake_llm import FakeLLMClient

from pacer.ai import LLMClient, load_prompt

PROMPT = load_prompt("connection_check")


def test_fake_answers_in_order_and_records_calls():
    fake: LLMClient = FakeLLMClient(answers=["ראשון", "שני"])
    assert fake.complete(PROMPT, "א").text == "ראשון"
    assert fake.complete(PROMPT, "ב").text == "שני"
    assert fake.complete(PROMPT, "ג").text == "שני"  # the last answer repeats
    assert [c.user_message for c in fake.calls] == ["א", "ב", "ג"]
    assert fake.calls[0].prompt.ref == "connection_check@v1"


def test_fake_can_answer_from_the_message():
    fake = FakeLLMClient(answers=lambda message: message.upper())
    assert fake.complete(PROMPT, "abc").text == "ABC"


def test_fake_also_refuses_a_call_without_a_prompt_file():
    with pytest.raises(TypeError):
        FakeLLMClient().complete("You are a helpful assistant", "שלום")
