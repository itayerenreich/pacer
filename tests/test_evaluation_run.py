"""Task 8: running the sets and scoring them, with fake models (no network, no cost).

The fake "model" answers by what it is asked: a judge's message, a library answer, a baseline, or a
talk summary. So every score here is known in advance, and every check is on the scoring itself.
"""

import json
from pathlib import Path

import pytest
from fake_llm import FakeLLMClient
from fakes import FakeUnitOfWork, empty_repos
from test_library_answer import CHUNKING, VectorEmbedder, add

from pacer.ai.anonymization import Anonymizer, load_roster
from pacer.ai.answer import NO_SOURCE, LibraryAnswerer
from pacer.ai.routing import AITask, ModelSpec, RoutingError
from pacer.demo import build_academy
from pacer.domain import NoteKind
from pacer.evaluation import LibraryCase, load_talk_set
from pacer.evaluation.judge import ALWAYS_CHECKED, grade_notes
from pacer.evaluation.report import render
from pacer.evaluation.run import (
    library_summary,
    router_for,
    run_library_case,
    run_talk_case,
    talk_summary,
    word_spans,
)
from pacer.evaluation.talk_set import resolve

TITLE = "משנה תורה, הלכות תלמוד תורה פרק ב'"


def answer_json(coverage="direct", passages=("P1",)) -> str:
    claims = [{"text": "עשרים וחמישה למלמד אחד.", "passages": list(passages)}] if passages else []
    return json.dumps(
        {"answer": 'ברמב"ם: עשרים וחמישה למלמד אחד.', "claims": claims, "coverage": coverage}, ensure_ascii=False
    )


def judge_json(items: int, claims: int, *, skip: int | None = None, helpful: int = 4) -> str:
    checklist = [{"item": i, "passed": True, "why": "כן"} for i in range(1, items + 1) if i != skip]
    verdicts = [{"claim": i, "supported": True, "why": "כן"} for i in range(1, claims + 1)]
    return json.dumps(
        {"checklist": checklist, "claims": verdicts, "helpful": helpful, "why": "טוב."}, ensure_ascii=False
    )


# אגם לנדאו's goals in the demo academy: 106 take part in group talks (0%), 107 fitness (50%)
T01_SUMMARY = {
    "summary": "[TRAINEE_35] דיברה פעמיים בשיחת הקבוצה ומתאמנת שלוש פעמים בשבוע.",
    "progress": [
        {"goal_id": 106, "progress": 30, "reason": "דיברה פעמיים בשיחת הקבוצה"},
        {"goal_id": 107, "progress": 90, "reason": "מתאמנת שלוש פעמים בשבוע"},  # out of the key's 55-75
    ],
    "notes": [
        {"kind": "worked", "text": "שעון מעורר רחוק מהמיטה עזר לה לקום"},
        {"kind": "observation", "text": "התנדבה לסחוב ציוד בטיול"},
    ],
}


def fake_model(message: str) -> str:
    if "CHECKLIST:" in message:  # the judge, grading an answer (with or without claims)
        claims = message.count("\n1. [") + message.count("\n2. [")  # claims are numbered with their ids
        items = message.split("CHECKLIST:")[1].split("PASSAGES:")[0].strip().count("\n") + 1
        return judge_json(items, claims)
    if "EXPECTED:" in message:  # the judge, matching notes: E1 -> N1, the rest not found
        expected = message.count("\nE") + 1
        matches = [{"expected": 1, "note": 1, "why": "אותו דבר"}] + [
            {"expected": i, "note": None, "why": "לא נמצא"} for i in range(2, expected + 1)
        ]
        return json.dumps({"matches": matches}, ensure_ascii=False)
    if "Passages:" in message:  # answering from the library
        return answer_json()
    if "The counselor's notes:" in message:
        return json.dumps(T01_SUMMARY, ensure_ascii=False)
    return json.dumps({"answer": "תשובה כללית, בלי מקורות מהספרייה."}, ensure_ascii=False)  # the baseline


@pytest.fixture
def repos():
    repos = empty_repos()
    build_academy(repos)
    return repos


@pytest.fixture
def cloud():
    return FakeLLMClient(answers=fake_model, runs_locally=False)


@pytest.fixture
def models(cloud):
    local = FakeLLMClient(answers=fake_model, model="fake/local", runs_locally=True)
    return {
        "claude-haiku": ModelSpec(runs_locally=False, make_client=lambda: cloud),
        "claude-sonnet": ModelSpec(runs_locally=False, make_client=lambda: cloud),
        "local-small": ModelSpec(runs_locally=True, make_client=lambda: local),
    }


@pytest.fixture
def router(models):
    return router_for("claude-haiku", models=models)


def case(**changes) -> LibraryCase:
    data = {
        "id": "L01",
        "question": "כמה תלמידים יכול מלמד אחד ללמד?",
        "kind": "about_source",
        "sources": [TITLE],
        "coverage": ["direct"],
        "judge": ["עשרים וחמישה למלמד"],
    }
    return LibraryCase.model_validate(data | changes)


def library_run(repos, router, the_case, *, embedder=None, **kwargs):
    uow = FakeUnitOfWork(repos)
    titles = {s.key: s.title for s in repos.sources.list_all()}
    answerer = LibraryAnswerer(uow, router, embedder or VectorEmbedder([1.0, 0.0]), CHUNKING)
    anonymizer = Anonymizer(load_roster(repos.staff, repos.trainees))
    return run_library_case(the_case, answerer, router, anonymizer, titles, **kwargs)


# --- the router of a run -------------------------------------------------------------------------------


def test_a_run_moves_only_the_evaluated_tasks_and_keeps_the_judge(models):
    router = router_for("local-small", models=models)
    assert router.route(AITask.ANSWER_FROM_LIBRARY).model == "local-small"
    assert router.route(AITask.SUMMARIZE_TALK).model == "local-small"
    assert router.route(AITask.JUDGE).model == "claude-sonnet"  # the same judge in every run
    assert router.route(AITask.EXTRACT_MEETING).model == "claude-haiku"
    with pytest.raises(RoutingError):
        router_for("gpt-unknown", models=models)


# --- the library ---------------------------------------------------------------------------------------


def test_a_library_case_is_answered_judged_and_compared_with_the_baseline(repos, router, cloud):
    add(repos, TITLE, [("עשרים וחמשה תינוקות למלמד אחד", [1.0, 0.0])])
    result = library_run(repos, router, case())

    assert result.coverage_ok and result.retrieved_right and result.cited_right
    assert result.grade.checklist == [True, True]  # the case's item, and the one always checked
    assert result.grade.claims == [True]
    assert result.baseline == "תשובה כללית, בלי מקורות מהספרייה." and result.baseline_grade.claims == []
    judge_messages = [c.user_message for c in cloud.calls if "CHECKLIST:" in c.user_message]
    assert len(judge_messages) == 2 and ALWAYS_CHECKED[0] in judge_messages[0]
    assert "PASSAGES:" in judge_messages[0] and "PASSAGES:" not in judge_messages[1]  # the baseline had none
    baseline_call = next(c for c in cloud.calls if c.prompt.ref == "library_baseline@v1")
    assert "Passages:" not in baseline_call.user_message


def test_no_relevant_passage_means_no_answer_model_and_an_honest_none(repos, router, cloud):
    add(repos, TITLE, [("רחוק מהשאלה", [0.0, 1.0])])
    off = case(id="L02", kind="off_topic", sources=[], coverage=["none"], judge=[])
    result = library_run(repos, router, off, embedder=VectorEmbedder([1.0, 0.0], floor=0.5), baseline=False)
    assert result.answer.answer == NO_SOURCE and result.coverage_ok
    assert result.retrieved_right is None and result.cited_right is None  # nothing to find
    assert not any(c.prompt.ref.startswith("library_answer") for c in cloud.calls)


def test_a_verdict_the_judge_skipped_is_none_never_a_pass(repos, models):
    add(repos, TITLE, [("עשרים וחמשה תינוקות למלמד אחד", [1.0, 0.0])])
    sloppy = FakeLLMClient(
        answers=lambda m: judge_json(2, 1, skip=2) if "CHECKLIST:" in m else fake_model(m), runs_locally=False
    )
    models = models | {"claude-sonnet": ModelSpec(runs_locally=False, make_client=lambda: sloppy)}
    result = library_run(repos, router_for("claude-haiku", models=models), case(), baseline=False)
    assert result.grade.checklist == [True, None] and result.grade.checklist_passed == 1


def test_the_library_summary(repos, router):
    add(repos, TITLE, [("עשרים וחמשה תינוקות למלמד אחד", [1.0, 0.0])])
    results = [library_run(repos, router, case()), library_run(repos, router, case(id="L02", reviewed=True))]
    summary = library_summary(results)
    assert summary["coverage_ok"] == 1.0 and summary["coverage_ok_reviewed"] == 1.0
    assert summary["retrieved_right"] == 1.0 and summary["claims_supported"] == 1.0
    assert summary["checklist"] == 1.0 and summary["helpful"] == 4 and summary["helpful_baseline"] == 4
    results[0].cost_usd = None
    assert library_summary(results)["cost_usd"] is None  # an unknown cost is not zero


# --- the talks -----------------------------------------------------------------------------------------


def t01(repos):
    talks = load_talk_set(Path("evals/talks-v1.json"))
    return resolve(talks.cases[0], repos.staff, repos.trainees, repos.goals, repos.groups)


def test_word_spans_find_whole_words_only():
    notes = "בן סיפר שהוא הבן הבכור. אחיו בן 15."
    from pacer.evaluation.talk_set import Mention

    spans = word_spans(notes, [Mention(word="בן", occurrence=2, means="word"), Mention(word="בן", means="trainee:x")])
    assert [notes[s:e] for s, e in spans] == ["בן"] and spans[0][0] == notes.index("בן 15")


def test_a_talk_is_scored_against_the_answer_key(repos, router):
    talk = t01(repos)
    roster = load_roster(repos.staff, repos.trainees)
    result = run_talk_case(talk, roster, router, words_stay_words=False)
    assert result.required_in_range == 1 and len(result.required) == 2
    assert result.wrong_progress == ["לשפר כושר לקראת יום המא״ה"]  # 90: above the key's range
    assert result.notes_found == 1 and result.notes_kind_right == 1  # E1 (worked) matched N1 (worked)
    assert result.forbidden_found == []


def test_the_two_variants_differ_only_in_the_marked_words(repos, router, cloud):
    talk = t01(repos)
    roster = load_roster(repos.staff, repos.trainees)
    run_talk_case(talk, roster, router, words_stay_words=False, judge=False)
    run_talk_case(talk, roster, router, words_stay_words=True, judge=False)
    masked, words = (c.user_message for c in cloud.calls if "The counselor's notes:" in c.user_message)
    assert "ל[TRAINEE_35]" in masked and "לאגם" not in masked
    assert "בטיול לאגם" in words and "השבוע [TRAINEE_35] דיברה" in words  # people still masked
    assert "היא כבר מתאמנת" in words and "היא כ[NAME_" in masked


def test_a_health_detail_in_the_summary_is_caught(repos, models):
    talk = t01(repos)
    leaky = dict(T01_SUMMARY, summary="[TRAINEE_35] סיפרה על הקרסול שלה ועל הרופא, ועל המסלולים.")
    client = FakeLLMClient(answers=[json.dumps(leaky, ensure_ascii=False)], runs_locally=False)
    models = models | {"claude-haiku": ModelSpec(runs_locally=False, make_client=lambda: client)}
    case_with_forbidden = talk.model_copy(
        update={"case": talk.case.model_copy(update={"forbidden": ["קרסול", "רופא"]})}
    )
    result = run_talk_case(
        case_with_forbidden,
        load_roster(repos.staff, repos.trainees),
        router_for("claude-haiku", models=models),
        words_stay_words=False,
        judge=False,
    )
    assert result.forbidden_found == ["קרסול", "רופא"]


def test_the_notes_judge_cannot_match_one_note_twice_or_a_note_that_does_not_exist(repos):
    answer = {
        "matches": [
            {"expected": 1, "note": 1, "why": "א"},
            {"expected": 2, "note": 1, "why": "ב"},
            {"expected": 3, "note": 9, "why": "ג"},
        ]
    }
    client = FakeLLMClient(answers=[json.dumps(answer, ensure_ascii=False)], runs_locally=False)
    grade = grade_notes(
        client,
        Anonymizer(load_roster(repos.staff, repos.trainees)),
        expected=[
            (NoteKind.WORKED, "שעון מעורר"),
            (NoteKind.FOLLOW_UP, "לחזור לשיחה"),
            (NoteKind.OBSERVATION, "סחבה ציוד"),
        ],
        proposed=[(NoteKind.WORKED, "שעון מעורר רחוק מהמיטה")],
    )
    assert grade.matches == [0, None, None]


def test_with_no_proposed_notes_the_judge_is_not_asked(repos):
    client = FakeLLMClient(answers=["not json"], runs_locally=False)
    grade = grade_notes(
        client, Anonymizer(load_roster(repos.staff, repos.trainees)), expected=[(NoteKind.WORKED, "משהו")], proposed=[]
    )
    assert grade.matches == [None] and client.calls == []


def test_the_talk_summary_and_the_report(repos, router):
    talk = t01(repos)
    roster = load_roster(repos.staff, repos.trainees)
    results = [run_talk_case(talk, roster, router, words_stay_words=w) for w in (False, True)]
    summary = talk_summary(results)
    assert set(summary) == {"masked", "words"} and summary["masked"]["required_in_range"] == 0.5
    assert summary["masked"]["wrong_progress"] == 1 and summary["masked"]["notes_found"] == 0.5
    text = render("הערכה", ["מטא"], None, (results, summary))
    assert "| מטרות חובה בטווח | 50% | 50% |" in text and "T01" in text


# --- the command line ----------------------------------------------------------------------------------


def test_run_writes_the_report_and_a_json_summary(repos, models, monkeypatch, tmp_path):
    import pacer.evaluation.__main__ as cli

    add(repos, TITLE, [("עשרים וחמשה תינוקות למלמד אחד", [1.0, 0.0])])
    monkeypatch.setattr(cli, "_uow", lambda: FakeUnitOfWork(repos))
    monkeypatch.setattr(cli, "router_for", lambda model: router_for(model, models=models))
    monkeypatch.setattr(cli, "ollama_embedder", lambda name: VectorEmbedder([1.0, 0.0]))
    monkeypatch.setattr(cli, "DEFAULT_CHUNKING", CHUNKING)
    out = tmp_path / "eval.md"
    cli.main(["run", "--limit", "1", "--out", str(out)])

    summary = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["model"] == "claude-haiku" and summary["judge"] == "claude-sonnet"
    assert summary["library"]["cases"] == 1 and set(summary["talks"]) == {"masked", "words"}
    assert summary["sets"] == ["library@v1", "talks@v1"]
    report = out.read_text(encoding="utf-8")
    assert report.startswith('<div dir="rtl">') and "## הספרייה" in report and "## שיחות אישיות" in report


def test_a_coverage_outside_the_acceptable_ones_is_counted_wrong(repos, models):
    add(repos, TITLE, [("עשרים וחמשה תינוקות למלמד אחד", [1.0, 0.0])])
    partial = FakeLLMClient(
        answers=lambda m: answer_json("partial") if "Passages:" in m else fake_model(m), runs_locally=False
    )
    models = models | {"claude-haiku": ModelSpec(runs_locally=False, make_client=lambda: partial)}
    result = library_run(repos, router_for("claude-haiku", models=models), case(), baseline=False, judge=False)
    assert result.answer.coverage == "partial" and not result.coverage_ok  # the case accepts only "direct"


def test_an_item_judged_twice_keeps_the_first_verdict(repos, models):
    add(repos, TITLE, [("עשרים וחמשה תינוקות למלמד אחד", [1.0, 0.0])])
    twice = json.dumps(
        {
            "checklist": [
                {"item": 1, "passed": True, "why": "א"},
                {"item": 1, "passed": False, "why": "ב"},
                {"item": 2, "passed": True, "why": "ג"},
            ],
            "claims": [],
            "helpful": 3,
            "why": "טוב.",
        },
        ensure_ascii=False,
    )
    judge = FakeLLMClient(answers=lambda m: twice if "CHECKLIST:" in m else fake_model(m), runs_locally=False)
    models = models | {"claude-sonnet": ModelSpec(runs_locally=False, make_client=lambda: judge)}
    result = library_run(repos, router_for("claude-haiku", models=models), case(), baseline=False)
    assert result.grade.checklist == [True, True] and result.grade.claims == [None]  # the claim never judged


def test_an_unknown_cost_makes_the_total_unknown():
    from pacer.evaluation.run import _total

    assert _total(0.1, 0.2) == pytest.approx(0.3)
    assert _total(0.1, None) is None
