"""Which model does which AI task, and the privacy rule that decides what is allowed.

Code asks for a TASK, never for a model:

    client = router.client_for(AITask.EXTRACT_MEETING)

and the routing table below decides which model serves it. One table means one place to
read the whole picture, and one place where privacy is enforced.

Privacy is enforced in CODE, not in prompts. Every task declares how sensitive the data
it sends is:

    PUBLIC      no real data (synthetic data, connection checks)     -> any model
    ANONYMIZED  text after anonymization (names replaced by tokens)   -> cloud is allowed
    PERSONAL    raw personal data                                     -> ONLY models on this computer

The router checks this three times: when the table is built (a bad table never starts),
when a client is handed out (the client itself must confirm it runs locally), and in a
test over the whole table (a bad change fails CI). There is deliberately NO fallback:
if the local model is down, a PERSONAL task fails ("fail closed"); it never quietly
moves to the cloud.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from types import MappingProxyType

from .clients import claude_client, ollama_client
from .llm_client import LLMClient


class Sensitivity(StrEnum):
    PUBLIC = "public"
    ANONYMIZED = "anonymized"
    PERSONAL = "personal"


class AITask(StrEnum):
    """Every AI task in Pacer. A new feature adds a member here and a row to ROUTES."""

    CONNECTION_CHECK = "connection_check"
    STRUCTURED_CHECK = "structured_check"
    EXTRACT_MEETING = "extract_meeting"  # task 6: summary + proposed tasks from a staff meeting
    SUMMARIZE_TALK = "summarize_talk"  # task 7: personal talk summary + proposed notes
    SYNTHETIC_DATA = "synthetic_data"  # stage 4: invented transcripts and cases for evaluation
    ANSWER_FROM_LIBRARY = "answer_from_library"  # stage 4: a staff question, answered from the library
    ANSWER_WITHOUT_LIBRARY = "answer_without_library"  # stage 4 evaluation: the same question, no passages
    JUDGE = "judge"  # stage 4 evaluation: grades an answer or a talk summary against the answer key
    SCREEN_RISK = "screen_risk"  # stage 5: is a counselor's question a risk situation? (after the phrase list)


@dataclass(frozen=True)
class ModelSpec:
    """A model Pacer may use. `runs_locally` is a promise checked against the real client."""

    runs_locally: bool
    make_client: Callable[[], LLMClient]


@dataclass(frozen=True)
class Route:
    model: str  # a key of the models table
    sensitivity: Sensitivity
    max_tokens: int = 1024


MODELS: Mapping[str, ModelSpec] = MappingProxyType(
    {
        "claude-haiku": ModelSpec(runs_locally=False, make_client=lambda: claude_client("claude-haiku-5-5")),
        "claude-sonnet": ModelSpec(
            runs_locally=False, make_client=lambda: claude_client("claude-sonnet-5-5")
        ),
        "local-small": ModelSpec(runs_locally=True, make_client=lambda: ollama_client("llama3.2")),
        # Stage 4, task 9: the open model compared with Claude. Chosen for Itay's laptop (16 GB, no GPU):
        # about 3 GB, multilingual, no "thinking" mode to slow every answer. ollama pull gemma3:4b
        "gemma3-4b": ModelSpec(runs_locally=True, make_client=lambda: ollama_client("gemma3:4b")),
    }
)

ROUTES: Mapping[AITask, Route] = MappingProxyType(
    {
        AITask.CONNECTION_CHECK: Route("claude-haiku", Sensitivity.PUBLIC, max_tokens=200),
        AITask.STRUCTURED_CHECK: Route("claude-haiku", Sensitivity.PUBLIC, max_tokens=500),
        AITask.EXTRACT_MEETING: Route("claude-haiku", Sensitivity.ANONYMIZED, max_tokens=4000),
        AITask.SUMMARIZE_TALK: Route("claude-haiku", Sensitivity.ANONYMIZED, max_tokens=4000),
        AITask.SYNTHETIC_DATA: Route("claude-sonnet", Sensitivity.PUBLIC, max_tokens=8000),
        # The library is not personal, but a question may name a trainee: it is anonymized first.
        AITask.ANSWER_FROM_LIBRARY: Route("claude-haiku", Sensitivity.ANONYMIZED, max_tokens=3000),
        # The evaluation. The baseline answers like ANSWER_FROM_LIBRARY, with the same model, minus the
        # passages. The judge is a stronger model than the one it grades (decision 2026-10-19).
        AITask.ANSWER_WITHOUT_LIBRARY: Route("claude-haiku", Sensitivity.ANONYMIZED, max_tokens=2000),
        AITask.JUDGE: Route("claude-sonnet", Sensitivity.ANONYMIZED, max_tokens=3000),
        # The safety screen sees exactly what the coach would send next: anonymized, health sentences removed.
        AITask.SCREEN_RISK: Route("claude-haiku", Sensitivity.ANONYMIZED, max_tokens=400),
    }
)


class RoutingError(Exception):
    """The routing table is wrong: a task without a route, or a route to an unknown model."""


class PrivacyViolation(Exception):
    """A task would send data to a model that is not allowed to see it. Never caught and ignored."""


def allowed(sensitivity: Sensitivity, *, runs_locally: bool) -> bool:
    """The whole privacy rule, in one line: personal data only on this computer."""
    return runs_locally or sensitivity is not Sensitivity.PERSONAL


class Router:
    """Hands out the right client for each task. Validates the whole table up front."""

    def __init__(
        self, routes: Mapping[AITask, Route] = ROUTES, models: Mapping[str, ModelSpec] = MODELS
    ) -> None:
        missing = [task for task in AITask if task not in routes]
        if missing:
            raise RoutingError(f"No route for: {', '.join(missing)}")
        for task, route in routes.items():
            spec = models.get(route.model)
            if spec is None:
                raise RoutingError(f"{task} routes to unknown model {route.model!r}")
            if not allowed(route.sensitivity, runs_locally=spec.runs_locally):
                raise PrivacyViolation(
                    f"{task} sends {route.sensitivity} data to {route.model}, which is not local"
                )
        self._routes = routes
        self._models = models
        self._clients: dict[str, LLMClient] = {}  # one client per model, created on first use

    def route(self, task: AITask) -> Route:
        return self._routes[task]

    def client_for(self, task: AITask) -> LLMClient:
        route = self._routes[task]
        client = self._clients.get(route.model)
        if client is None:
            client = self._models[route.model].make_client()
            self._clients[route.model] = client
        # Second check, on the real client: a table entry that says "local" is a promise;
        # the client is the truth (e.g. someone pointed "local-small" at a cloud model).
        if client.runs_locally != self._models[route.model].runs_locally:
            raise RoutingError(
                f"Model {route.model!r} is declared "
                f"{'local' if self._models[route.model].runs_locally else 'cloud'}, but its client is not"
            )
        if not allowed(route.sensitivity, runs_locally=client.runs_locally):
            raise PrivacyViolation(
                f"{task} sends {route.sensitivity} data to {client.model}, which is not local"
            )
        return client


@cache
def get_router() -> Router:
    """The production router, built (and validated) once per process. Tests build their own."""
    return Router()
