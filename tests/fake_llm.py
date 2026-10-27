"""FakeLLMClient: a stand-in for a real model in tests. No network, no cost, same answer every time.

It satisfies the LLMClient Protocol, so any code that accepts an LLMClient accepts it.
It also records every call, so a test can check WHAT was sent: which prompt version,
and that no personal data reached a model that is not local.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pacer.ai import LLMResponse, Prompt
from pacer.ai.llm_client import ChatResponse, ToolCall
from pacer.ai.routing import ModelSpec, Router


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
    # A function may also raise, to simulate a provider error (e.g. LLMUnavailableError).
    model: str = "fake/model"
    runs_locally: bool = True
    native_schema: bool = False  # True: like Claude, the schema goes to the "provider", not the message
    calls: list[Call] = field(default_factory=list)

    truncated: bool = False  # set True to simulate an answer cut off at max_tokens

    # The agent (stage 6): what chat() answers, in order (the last one repeats), or a function of the
    # messages so far. Each is a Reply: say("text") or use("tool", arg=...). Every chat is recorded,
    # with a copy of the messages AS SENT, so a test can check exactly what the model read.
    replies: "list[Reply] | Callable[[list[dict[str, Any]]], Reply]" = field(default_factory=list)
    chats: "list[ChatCall]" = field(default_factory=list)
    before_chat: Callable[[], None] | None = None  # e.g. assert that no transaction is open

    def chat(self, prompt, messages, *, tools, max_tokens=1024, allow_tools=True) -> ChatResponse:
        import copy

        if not isinstance(prompt, Prompt):
            raise TypeError("chat() needs a Prompt loaded from a prompt file")
        if self.before_chat is not None:
            self.before_chat()
        self.chats.append(ChatCall(prompt, copy.deepcopy(list(messages)), list(tools), max_tokens, allow_tools))
        if callable(self.replies):
            reply = self.replies(list(messages))
        elif not self.replies:
            reply = say("תשובה לדוגמה")
        else:
            reply = self.replies[min(len(self.chats), len(self.replies)) - 1]
        n = len(self.chats)
        calls = tuple(
            ToolCall(id=f"call_{n}_{i}", name=name, arguments=args) for i, (name, args) in enumerate(reply.calls)
        )
        return ChatResponse(
            text=reply.text,
            tool_calls=calls,
            model=self.model,
            prompt_ref=prompt.ref,
            input_tokens=sum(len(str(m.get("content") or "").split()) for m in messages),
            output_tokens=len(reply.text.split()),
            cost_usd=reply.cost_usd,
            latency_ms=0,
            truncated=reply.truncated,
        )

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


@dataclass(frozen=True)
class Reply:
    text: str = ""
    calls: tuple[tuple[str, str], ...] = ()  # (tool name, JSON arguments)
    cost_usd: float | None = 0.0
    truncated: bool = False


def say(text: str, **kwargs) -> Reply:
    return Reply(text=text, **kwargs)


def use(*calls: "tuple[str, dict[str, Any]] | str", text: str = "", **kwargs) -> Reply:
    """use(("find_trainees", {"name": "[NAME_1]"}), ("attendance", {})) or use("week_schedule")."""
    import json

    pairs = []
    for call in calls:
        name, args = (call, {}) if isinstance(call, str) else call
        pairs.append((name, args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)))
    return Reply(text=text, calls=tuple(pairs), **kwargs)


@dataclass
class ChatCall:
    prompt: Prompt
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    max_tokens: int
    allow_tools: bool

    def tool_results(self) -> list[str]:
        return [m["content"] for m in self.messages if m["role"] == "tool"]

    def everything(self) -> str:
        """Every text the model read in this call: the messages, and the arguments it wrote before."""
        parts = []
        for m in self.messages:
            parts.append(str(m.get("content") or ""))
            parts += [c["function"]["arguments"] for c in m.get("tool_calls", [])]
        return "\n".join(parts)


def fake_router(cloud: FakeLLMClient, local: FakeLLMClient | None = None) -> Router:
    """The real Router (with all its privacy checks), with fakes in place of the real models.

    `cloud` must say runs_locally=False, as Claude would: the router checks every client against
    the table, so a fake cannot pretend to be local and slip past the privacy rule.
    """
    local = local or FakeLLMClient(model="fake/local", runs_locally=True)
    return Router(
        models={
            "claude-haiku": ModelSpec(runs_locally=False, make_client=lambda: cloud),
            "claude-sonnet": ModelSpec(runs_locally=False, make_client=lambda: cloud),
            "local-small": ModelSpec(runs_locally=True, make_client=lambda: local),
        }
    )


class FakeEmbedder:
    """Embeddings without a model: each word adds 1 to one of 1024 positions (by a fixed hash).
    Texts that share words get close vectors, texts that share none are far apart. Deterministic,
    instant, and enough to test indexing and search. Records what it embedded."""

    def __init__(self, name: str = "fake-embed", *, runs_locally: bool = True, dimensions: int = 1024) -> None:
        from pacer.ai.embeddings import EmbeddingSpec

        self.spec = EmbeddingSpec(name=name, litellm_model=f"fake/{name}", dimensions=dimensions)
        self.runs_locally = runs_locally
        self.documents: list[str] = []
        self.queries: list[str] = []
        self.calls = 0

    def _vector(self, text: str) -> list[float]:
        import zlib

        vector = [0.0] * self.spec.dimensions
        for word in text.split():
            vector[zlib.crc32(word.strip(".,:;!?\"'״׳").encode()) % self.spec.dimensions] += 1.0
        if not any(vector):
            vector[0] = 1e-6  # never a zero vector
        return vector

    def embed_documents(self, texts) -> list[list[float]]:
        self.calls += 1
        self.documents += list(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)
