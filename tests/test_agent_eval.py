"""The agent's evaluation set fits the demo academy, and its checks score what they say they score."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fakes import empty_repos

from pacer.ai.agent import AgentOutcome, AgentResult, AgentStep
from pacer.ai.tools import TOOL_NAMES
from pacer.demo import build_academy
from pacer.evaluation.agent_eval import (
    AgentCase,
    AgentCaseResult,
    AgentEvalSet,
    ResolvedAgentCase,
    agent_summary,
    load_agent_set,
    render_agent,
    resolve_agent_cases,
)
from pacer.services.permissions import can_view_trainee

AGENT_SET = Path(__file__).parent.parent / "evals" / "agent-v1.json"


@pytest.fixture(scope="module")
def academy():
    repos = empty_repos()
    build_academy(repos, as_of=datetime(2026, 10, 27, tzinfo=UTC), schedule=False)
    return repos


def test_the_set_fits_the_demo_academy_and_the_tools(academy):
    eval_set = load_agent_set(AGENT_SET)
    resolved, problems = resolve_agent_cases(eval_set, academy, TOOL_NAMES)
    assert problems == [] and len(resolved) == len(eval_set.cases)


def test_every_trainee_a_case_is_about_is_visible_to_its_asker_except_the_permission_case(academy):
    for item in resolve_agent_cases(load_agent_set(AGENT_SET), academy, TOOL_NAMES)[0]:
        asker = academy.staff.get(item.asker_id)
        for trainee_id in item.about_ids:
            group = academy.groups.get(academy.trainees.get(trainee_id).group_id)
            assert can_view_trainee(asker, group), item.case.id


def test_the_facts_the_set_relies_on_are_true_of_the_demo(academy):
    """Two trainees named נדב in group ארז and one in אלון (A02), so the ambiguity is real."""
    nadavs = [t for t in academy.trainees.list_all() if t.first_name == "נדב"]
    by_group = sorted((academy.groups.get(t.group_id).name, t.last_name) for t in nadavs)
    assert by_group == [("קבוצת אלון", "אשכנזי"), ("קבוצת ארז", "הררי"), ("קבוצת ארז", "נחום")]


def test_the_set_covers_the_three_outcomes_and_the_definition_of_done():
    cases = load_agent_set(AGENT_SET).cases
    assert {c.outcome for c in cases} == {"answer", "referral", "not_sent"}
    assert any("עומד ביעדים" in c.message and "ask_coach" in c.must_call for c in cases)


def test_a_typo_is_reported_not_skipped(academy):
    broken = AgentEvalSet(
        name="t",
        version=1,
        about="x",
        cases=[
            AgentCase(
                id="A01",
                tests="x",
                asker="נועה לוי",
                message="שאלה ארוכה",
                outcome="answer",
                must_call=["ask_coahc"],
                about=["נדב הררר"],
            )
        ],
    )
    _, problems = resolve_agent_cases(broken, academy, TOOL_NAMES)
    assert problems == ["A01: no tool named ask_coahc", "A01: no trainee named נדב הררר"]


# --- the scoring ---------------------------------------------------------------------------------


def scored(case: AgentCase, steps, answer="תשובה", outcome=AgentOutcome.ANSWER, about=frozenset()):
    result = AgentResult(outcome, answer=answer, steps=[AgentStep(t, a, True, "s", 1) for t, a in steps])
    return AgentCaseResult(ResolvedAgentCase(case, 2, about), result, 1.0)


CASE = AgentCase(
    id="A01",
    tests="x",
    asker="נועה לוי",
    message="שאלה ארוכה",
    outcome="answer",
    must_call=["ask_coach"],
    any_of=[["trainee_record", "attendance"]],
    must_not_call=["report_exceptional_incident"],
    about=["נדב נחום"],
    answer_mentions=["נחום"],
    answer_must_not=["הררי"],
)


def test_the_right_tools_about_the_right_trainee_pass():
    r = scored(
        CASE, [("attendance", {"trainee_id": 15}), ("ask_coach", {"trainee_id": 15})], "נדב נחום", about=frozenset({15})
    )
    assert (r.outcome_ok, r.tools_ok, r.trainee_ok, r.text_ok, r.passed) == (True, True, True, True, True)


@pytest.mark.parametrize(
    ("steps", "answer", "failed"),
    [
        ([("ask_coach", {"trainee_id": 15})], "נחום", "tools"),  # none of the any_of group
        ([("attendance", {"trainee_id": 15})], "נחום", "tools"),  # no ask_coach
        (
            [
                ("attendance", {"trainee_id": 15}),
                ("ask_coach", {"trainee_id": 15}),
                ("report_exceptional_incident", {}),
            ],
            "נחום",
            "tools",
        ),
        (
            [("attendance", {"trainee_id": 8}), ("ask_coach", {"trainee_id": 15})],
            "נחום",
            "trainee",
        ),  # one call about another
        (
            [("trainee_record", {"trainee_id": 15}), ("ask_coach", {"trainee_id": 15})],
            "נדב",
            "text",
        ),  # does not mention
        ([("trainee_record", {"trainee_id": 15}), ("ask_coach", {"trainee_id": 15})], "נחום ולא הררי", "text"),
    ],
)
def test_each_check_fails_on_its_own(steps, answer, failed):
    r = scored(CASE, steps, answer, about=frozenset({15}))
    checks = {"tools": r.tools_ok, "trainee": r.trainee_ok, "text": r.text_ok}
    assert checks.pop(failed) is False and all(v is not False for v in checks.values())
    assert r.passed is False


def test_a_case_about_a_trainee_fails_if_no_call_is_about_anyone():
    r = scored(CASE.model_copy(update={"must_call": [], "any_of": []}), [], "נחום", about=frozenset({15}))
    assert r.trainee_ok is False


def test_checks_the_case_does_not_ask_for_are_not_counted():
    bare = AgentCase(id="A02", tests="x", asker="נועה לוי", message="שאלה ארוכה", outcome="referral")
    r = scored(bare, [], None, AgentOutcome.REFERRAL)
    assert (r.trainee_ok, r.text_ok, r.passed) == (None, None, True)
    summary = agent_summary([r])
    assert summary["trainee_ok"] is None and summary["passed"] == 1.0 and summary["tool_calls_per_answer"] is None


def test_the_report_renders():
    r = scored(CASE, [("ask_coach", {"trainee_id": 15})], "נחום", about=frozenset({15}))
    report = render_agent([r], agent_summary([r]), ["meta"])
    assert "# הערכת הסוכן" in report and "`ask_coach`" in report and "נחום" in report


def test_a_call_that_failed_is_about_no_one():
    result = AgentResult(
        AgentOutcome.ANSWER,
        answer="נחום",
        steps=[
            AgentStep("trainee_record", {"trainee_id": 8}, False, "לא נמצא", 1),
            AgentStep("ask_coach", {"trainee_id": 15}, True, "s", 1),
        ],
    )
    r = AgentCaseResult(ResolvedAgentCase(CASE, 2, frozenset({15})), result, 1.0)
    assert r.trainee_ok is True


def test_zero_tool_calls_per_answer_is_shown_as_zero():
    bare = AgentCase(id="A02", tests="x", asker="נועה לוי", message="שאלה ארוכה", outcome="answer")
    r = scored(bare, [], "תשובה")
    assert "| קריאות tools לתשובה | 0.0 |" in render_agent([r], agent_summary([r]), [])
