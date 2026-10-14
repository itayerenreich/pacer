"""The router: the production table is safe, and every way of breaking privacy is caught."""

import pytest
from fake_llm import FakeLLMClient

from pacer.ai import LLMUnavailableError, load_prompt
from pacer.ai.routing import (
    MODELS,
    ROUTES,
    AITask,
    ModelSpec,
    PrivacyViolation,
    Route,
    Router,
    RoutingError,
    Sensitivity,
    allowed,
)

PUBLIC, ANONYMIZED, PERSONAL = Sensitivity.PUBLIC, Sensitivity.ANONYMIZED, Sensitivity.PERSONAL


def models(**specs: ModelSpec) -> dict[str, ModelSpec]:
    return specs


def local_model() -> ModelSpec:
    return ModelSpec(
        runs_locally=True, make_client=lambda: FakeLLMClient(model="fake/local", runs_locally=True)
    )


def cloud_model() -> ModelSpec:
    return ModelSpec(
        runs_locally=False, make_client=lambda: FakeLLMClient(model="fake/cloud", runs_locally=False)
    )


def routes_all_to(model: str, sensitivity: Sensitivity = PUBLIC, **overrides: Route) -> dict[AITask, Route]:
    table = {task: Route(model, sensitivity) for task in AITask}
    table.update({AITask(name): route for name, route in overrides.items()})
    return table


# --- the production table ---------------------------------------------------------------


def test_every_task_has_a_route_to_a_known_model():
    for task in AITask:
        assert ROUTES[task].model in MODELS, task


def test_no_personal_task_reaches_a_cloud_model():
    """The CI guard: a change that routes personal data to the cloud fails here."""
    for task, route in ROUTES.items():
        assert allowed(route.sensitivity, runs_locally=MODELS[route.model].runs_locally), task


def test_the_production_router_builds():
    Router()  # validates the whole table


def test_the_tables_cannot_be_changed_at_runtime():
    with pytest.raises(TypeError):
        ROUTES[AITask.EXTRACT_MEETING] = Route("claude-haiku", PUBLIC)  # type: ignore[index]


def test_meeting_extraction_is_anonymized_on_haiku():
    """The decisions of 2026-10-14, as executable facts."""
    assert ROUTES[AITask.EXTRACT_MEETING] == Route("claude-haiku", ANONYMIZED, max_tokens=4000)


# --- the privacy rule -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sensitivity", "runs_locally", "expected"),
    [
        (PUBLIC, True, True),
        (PUBLIC, False, True),
        (ANONYMIZED, True, True),
        (ANONYMIZED, False, True),
        (PERSONAL, True, True),
        (PERSONAL, False, False),  # the only forbidden combination
    ],
)
def test_privacy_rule(sensitivity, runs_locally, expected):
    assert allowed(sensitivity, runs_locally=runs_locally) is expected


def test_a_table_sending_personal_data_to_the_cloud_never_starts():
    table = routes_all_to("cloud", summarize_talk=Route("cloud", PERSONAL))
    with pytest.raises(PrivacyViolation, match="summarize_talk sends personal data to cloud"):
        Router(table, models(cloud=cloud_model()))


def test_personal_data_on_a_local_model_is_fine():
    table = routes_all_to("cloud", summarize_talk=Route("local", PERSONAL))
    router = Router(table, models(cloud=cloud_model(), local=local_model()))
    assert router.client_for(AITask.SUMMARIZE_TALK).runs_locally


def test_a_model_declared_local_whose_client_is_cloud_is_caught():
    """Someone points the 'local' entry at a cloud model: the client tells the truth."""
    liar = ModelSpec(runs_locally=True, make_client=lambda: FakeLLMClient(runs_locally=False))
    table = routes_all_to("cloud", summarize_talk=Route("liar", PERSONAL))
    router = Router(table, models(cloud=cloud_model(), liar=liar))
    with pytest.raises(RoutingError, match="declared local"):
        router.client_for(AITask.SUMMARIZE_TALK)


# --- table mistakes -----------------------------------------------------------------------


def test_a_task_without_a_route_is_an_error():
    table = routes_all_to("cloud")
    del table[AITask.SYNTHETIC_DATA]
    with pytest.raises(RoutingError, match="No route for: synthetic_data"):
        Router(table, models(cloud=cloud_model()))


def test_a_route_to_an_unknown_model_is_an_error():
    with pytest.raises(RoutingError, match="unknown model 'gpt-9'"):
        Router(routes_all_to("gpt-9"), models(cloud=cloud_model()))


# --- handing out clients --------------------------------------------------------------------


def test_one_client_per_model_reused_across_tasks():
    built = []

    def make():
        built.append(1)
        return FakeLLMClient(runs_locally=False)

    router = Router(routes_all_to("cloud"), models(cloud=ModelSpec(runs_locally=False, make_client=make)))
    assert router.client_for(AITask.EXTRACT_MEETING) is router.client_for(AITask.SUMMARIZE_TALK)
    assert len(built) == 1


def test_clients_are_built_lazily():
    """Building the router does not build clients: no key is needed until a cloud task runs."""
    built = []
    spec = ModelSpec(
        runs_locally=False, make_client=lambda: built.append(1) or FakeLLMClient(runs_locally=False)
    )
    Router(routes_all_to("cloud"), models(cloud=spec))
    assert built == []


def test_fail_closed_no_fallback_to_the_cloud():
    """The local model is down: a personal task fails. The cloud model is never even built."""

    class DownLocal(FakeLLMClient):
        def complete(self, *args, **kwargs):
            raise LLMUnavailableError("Ollama is not running")

    cloud_built = []
    table = routes_all_to("cloud", summarize_talk=Route("local", PERSONAL))
    router = Router(
        table,
        models(
            cloud=ModelSpec(
                runs_locally=False,
                make_client=lambda: cloud_built.append(1) or FakeLLMClient(runs_locally=False),
            ),
            local=ModelSpec(runs_locally=True, make_client=lambda: DownLocal(runs_locally=True)),
        ),
    )
    with pytest.raises(LLMUnavailableError):
        router.client_for(AITask.SUMMARIZE_TALK).complete(load_prompt("connection_check"), "שלום")
    assert cloud_built == []
