"""The coach's evaluation set: valid, fits the demo academy, and every asker may see their trainee."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from fakes import empty_repos

from pacer.demo import build_academy
from pacer.domain import TraineeStatus
from pacer.evaluation.coach_eval import CoachCase, load_coach_set, resolve_cases
from pacer.services.permissions import can_view_trainee

COACH_SET = Path(__file__).parent.parent / "evals" / "coach-v1.json"


@pytest.fixture(scope="module")
def academy():
    repos = empty_repos()
    build_academy(repos, as_of=datetime(2026, 10, 25, tzinfo=UTC))
    return repos


def test_the_set_fits_the_demo_academy(academy):
    resolved, problems = resolve_cases(load_coach_set(COACH_SET), academy)
    assert problems == []
    assert len(resolved) == len(load_coach_set(COACH_SET).cases)


def test_every_asker_may_see_their_trainee_and_every_trainee_is_active(academy):
    for item in resolve_cases(load_coach_set(COACH_SET), academy)[0]:
        if item.trainee_id is None:
            continue
        trainee = academy.trainees.get(item.trainee_id)
        assert trainee.status is TraineeStatus.ACTIVE, item.case.id
        asker = academy.staff.get(item.asker_id)
        assert can_view_trainee(asker, academy.groups.get(trainee.group_id)), item.case.id


def test_the_set_has_a_general_question_and_an_exceptional_incident():
    cases = load_coach_set(COACH_SET).cases
    assert any(c.trainee is None for c in cases)
    assert any(c.outcome == "referral" for c in cases)


def test_an_advice_case_needs_a_checklist_and_a_referral_has_none():
    with pytest.raises(ValueError):
        CoachCase(id="C99", tests="x", trainee=None, asker="x", question="שאלה ארוכה", outcome="advice", checklist=[])
    with pytest.raises(ValueError):
        CoachCase(id="C99", tests="x", trainee=None, asker="x", question="שאלה ארוכה", outcome="referral", checklist=["x"])


def test_a_typo_in_a_name_is_reported_not_skipped(academy):
    from pacer.evaluation.coach_eval import CoachEvalSet

    broken = CoachEvalSet(
        name="t", version=1, about="x",
        cases=[CoachCase(id="C01", tests="x", trainee="נדב הררר", asker="עומר כהן", question="שאלה ארוכה", outcome="advice", checklist=["x"])],
    )
    resolved, problems = resolve_cases(broken, academy)
    assert resolved == [] and problems == ["C01: no trainee named נדב הררר"]


def test_a_whole_run_with_fake_models_scores_and_reports(academy):
    """The path the paid command takes, end to end, with fakes: the coach with and without the record,
    the judge, the summary and the report. The fake judge passes every item when the answer used the
    record (it says "נקודה מראש"), and fails them otherwise: so the two numbers must differ."""
    import json

    from fake_llm import FakeLLMClient, fake_router
    from fakes import FakeUnitOfWork
    from test_library_answer import CHUNKING, VectorEmbedder

    from pacer.ai.anonymization import Anonymizer, load_roster
    from pacer.ai.coach import Coach
    from pacer.ai.routing import AITask
    from pacer.evaluation.coach_eval import coach_summary, render_coach, run_coach_case

    def answer(message: str) -> str:
        if "CHECKLIST:" in message:
            used = "נקודה מראש" in message
            count = message.split("CHECKLIST:")[1].count("\n")
            return json.dumps({
                "checklist": [{"item": i, "passed": used, "why": "x"} for i in range(1, count + 1)],
                "claims": [], "helpful": 4 if used else 2, "why": "x",
            })
        if "Trainee:" in message:
            action = "לבקש ממנו שוב שיכין נקודה מראש" if "[worked]" in message else "שיחה אישית קצרה"
            return json.dumps({"risk": False, "understanding": "מובן מה קורה כאן.", "questions": [],
                               "steps": [{"action": action, "example": None, "passages": []}], "involve": []},
                              ensure_ascii=False)
        return json.dumps({"risk": False, "kinds": [], "reason": "nothing"})

    router = fake_router(FakeLLMClient(answers=answer, runs_locally=False))
    coach = Coach(FakeUnitOfWork(academy), router, VectorEmbedder([1.0, 0.0], floor=0.4), CHUNKING)
    resolved, _ = resolve_cases(load_coach_set(COACH_SET), academy)
    anonymizer = Anonymizer(load_roster(academy.staff, academy.trainees))
    actors = {s.key: s for s in academy.staff.list_all()}
    judge = router.client_for(AITask.JUDGE)
    results = [run_coach_case(item, coach, actors, judge, anonymizer, baseline=True) for item in resolved]

    summary = coach_summary(results)
    assert summary["outcome_ok"] == 1.0  # C09 is a referral, by the phrase list or the screen
    assert summary["checklist_personal"] > summary["checklist_personal_without_record"]
    report = render_coach(results, summary, ["meta"])
    assert "## C01" in report and "| סעיף | עם הרשומה | בלי |" in report


def _graded(checklist: list[str], verdicts: list, baseline: list | None = None):
    from pacer.ai.coach import CoachOutcome, CoachResult
    from pacer.evaluation.coach_eval import CoachCaseResult
    from pacer.evaluation.judge import Grade

    case = CoachCase(id="C01", tests="x", trainee="t", asker="a", question="שאלה ארוכה", outcome="advice", checklist=checklist)
    result = CoachResult(CoachOutcome.ADVICE, "q", 1)
    out = CoachCaseResult(case, result, 1.0, grade=Grade(checklist=verdicts, claims=[True, None], helpful=3, why="", cost_usd=0.0))
    if baseline is not None:
        out.baseline = result
        out.baseline_grade = Grade(checklist=baseline, claims=[], helpful=2, why="", cost_usd=0.0)
    return out


def test_the_always_checked_item_is_not_part_of_the_personal_score():
    """The review: a baseline that failed all 3 personal items scored 25%, from "no diagnosis"."""
    from pacer.evaluation.coach_eval import coach_summary

    summary = coach_summary([_graded(["a", "b", "c"], [True, True, True, True], baseline=[False, False, False, True])])
    assert summary["checklist_personal"] == 1.0 and summary["checklist_personal_without_record"] == 0.0
    assert summary["always_checked"] == 1.0


def test_a_skipped_verdict_is_neither_a_pass_nor_a_fail():
    from pacer.evaluation.coach_eval import coach_summary

    summary = coach_summary([_graded(["a", "b", "c"], [True, None, True, True])])
    assert summary["checklist"] == 1.0 and summary["skipped_verdicts"] == 1
    assert summary["claims_supported"] == 1.0


def test_the_judge_never_receives_a_health_sentence():
    from pacer.ai.anonymization import Anonymizer, Roster
    from pacer.evaluation.judge import answer_message

    message = answer_message(Anonymizer(Roster([])), question="הוא לוקח ריטלין. איך לעזור?", answer="תשובה", checklist=["x"])
    assert "ריטלין" not in message and "איך לעזור" in message


def test_a_judge_failure_keeps_the_coachs_answer(academy):
    """C06: the judge failed, and the whole case was dropped, though the coach had answered."""
    import json

    from fake_llm import FakeLLMClient, fake_router
    from fakes import FakeUnitOfWork
    from test_library_answer import CHUNKING, VectorEmbedder

    from pacer.ai import LLMUnavailableError
    from pacer.ai.anonymization import Anonymizer, load_roster
    from pacer.ai.coach import Coach, CoachOutcome
    from pacer.evaluation.coach_eval import coach_summary, run_coach_case

    def answer(message: str) -> str:
        if "CHECKLIST:" in message:
            raise LLMUnavailableError("the judge is down")
        if "Trainee:" in message:
            return json.dumps({"risk": False, "understanding": "מובן מה קורה כאן.", "questions": [],
                               "steps": [{"action": "שיחה אישית קצרה", "example": None, "passages": []}],
                               "involve": []}, ensure_ascii=False)
        return json.dumps({"risk": False, "kinds": [], "reason": "nothing"})

    router = fake_router(FakeLLMClient(answers=answer, runs_locally=False))
    from pacer.ai.routing import AITask

    coach = Coach(FakeUnitOfWork(academy), router, VectorEmbedder([1.0, 0.0], floor=0.4), CHUNKING)
    item = resolve_cases(load_coach_set(COACH_SET), academy)[0][0]
    out = run_coach_case(
        item, coach, {s.key: s for s in academy.staff.list_all()}, router.client_for(AITask.JUDGE),
        Anonymizer(load_roster(academy.staff, academy.trainees)), baseline=True,
    )
    assert out.result.outcome is CoachOutcome.ADVICE and out.grade is None
    assert out.notes == ["judge failed: LLMUnavailableError"]
    assert coach_summary([out])["judge_failed"] == 1
