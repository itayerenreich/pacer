"""The evaluation sets (evals/) and the masking measurement. No model, no database: the demo academy in memory.

The sets are data, and data can be wrong like code: these tests check that every case is consistent
with itself and fits the academy, so a typo fails here instead of quietly counting as a miss.
"""

import json
from pathlib import Path

import pytest
from fakes import FakeUnitOfWork, empty_repos
from pydantic import ValidationError

from pacer.ai.anonymization import load_roster
from pacer.demo import build_academy
from pacer.evaluation import CaseKind, LibraryCase, load_library_set, load_talk_set, unknown_sources
from pacer.evaluation.masking import Outcome, classify, measure, summary
from pacer.evaluation.talk_set import CaseError, Mention, TalkCase, resolve
from pacer.library.manifest import load_manifest

LIBRARY_SET = Path("evals/library-v1.json")
TALK_SET = Path("evals/talks-v1.json")


@pytest.fixture(scope="module")
def repos():
    repos = empty_repos()
    build_academy(repos)
    return repos


@pytest.fixture(scope="module")
def roster(repos):
    return load_roster(repos.staff, repos.trainees)


# --- the library set ----------------------------------------------------------------------------------


def test_the_library_set_is_valid_and_every_source_exists():
    eval_set = load_library_set(LIBRARY_SET)
    titles = {s.title for s in load_manifest(Path("library/classics.json")).sources}
    assert eval_set.ref == "library@v1" and len(eval_set.cases) >= 20
    assert unknown_sources(eval_set, titles) == []
    assert {c.kind for c in eval_set.cases} == set(CaseKind)  # every kind is represented


def case(**changes) -> dict:
    return {
        "id": "L01",
        "question": "שאלה כלשהי?",
        "kind": "applied",
        "sources": ["משלי"],
        "coverage": ["partial"],
    } | changes


def test_a_library_case_must_be_consistent():
    LibraryCase.model_validate(case())
    LibraryCase.model_validate(case(kind="safety", sources=[], coverage=["none"]))  # safety: with or without
    LibraryCase.model_validate(case(kind="safety"))
    bad = [
        case(coverage=["none"]),  # a source, yet "none"
        case(coverage=["partial", "none"]),  # "none" never mixes with the others
        case(sources=[], coverage=["partial"]),  # no source, yet covered
        case(kind="off_topic"),  # off-topic with a source
        case(kind="hard_negative", sources=[], coverage=["direct"]),
        case(sources=[], coverage=["none"]),  # an applied case always has a source
        case(sources=["משלי", "משלי"]),
        case(id="Q1"),
    ]
    for data in bad:
        with pytest.raises(ValidationError):
            LibraryCase.model_validate(data)


def test_two_cases_with_the_same_question_are_refused(tmp_path):
    data = {"name": "x", "version": 1, "about": "x", "cases": [case(), case(id="L02")]}
    path = tmp_path / "x.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValidationError, match="same question"):
        load_library_set(path)


def test_unknown_sources_names_the_typo():
    eval_set = load_library_set(LIBRARY_SET)
    titles = {s for c in eval_set.cases for s in c.sources}
    assert unknown_sources(eval_set, titles - {'איגרת הרמב"ן'}) == ['איגרת הרמב"ן']


# --- the talk set -------------------------------------------------------------------------------------


def test_every_talk_case_fits_the_demo_academy(repos):
    talks = load_talk_set(TALK_SET)
    assert talks.ref == "talks@v1" and len(talks.cases) >= 8
    for c in talks.cases:
        resolve(c, repos.staff, repos.trainees, repos.goals, repos.groups)  # raises CaseError if not


def talk(**changes) -> TalkCase:
    data = {
        "id": "T01",
        "tests": "test",
        "trainee": "אגם לנדאו",
        "counselor": "איתי ביטון",
        "held_on": "2026-10-14",
        "notes": "שיחה עם אגם. בטיול לאגם היא סחבה ציוד.",
        "mentions": [{"word": "לאגם", "means": "word"}],
    }
    return TalkCase.model_validate(data | changes)


def test_a_mention_must_be_in_the_notes_and_a_case_must_fit_the_academy(repos):
    with pytest.raises(ValidationError, match="does not appear 2"):
        talk(mentions=[{"word": "לאגם", "occurrence": 2, "means": "word"}])
    with pytest.raises(ValidationError):
        talk(mentions=[{"word": "לאגם", "means": "friend"}])  # not word / trainee: / staff:
    with pytest.raises(CaseError, match="no trainee named"):
        resolve(talk(trainee="אין כזו"), repos.staff, repos.trainees, repos.goals)
    with pytest.raises(CaseError, match="has no goal"):
        resolve(talk(progress=[{"goal": "לטוס לירח", "low": 0, "high": 10}]), repos.staff, repos.trainees, repos.goals)
    with pytest.raises(CaseError, match="not in the academy"):
        resolve(
            talk(mentions=[{"word": "אגם", "means": "trainee:אגם מהשכונה"}]), repos.staff, repos.trainees, repos.goals
        )
    goals = repos.goals
    with pytest.raises(CaseError, match="is left, not active"):  # אגם אוחיון left the academy
        resolve(talk(trainee="אגם אוחיון", counselor="נועה לוי"), repos.staff, repos.trainees, goals, repos.groups)
    with pytest.raises(CaseError, match="not a counselor of"):
        resolve(talk(counselor="נועה לוי"), repos.staff, repos.trainees, goals, repos.groups)
    dropped = {"goal": "להכיר לעומק כל אחד מחברי הקבוצה", "low": 30, "high": 60}  # רעות קליין's, dropped
    with pytest.raises(CaseError, match="is dropped: a talk cannot move it"):
        resolve(
            talk(trainee="רעות קליין", counselor="נועה לוי", progress=[dropped]), repos.staff, repos.trainees, goals
        )
    with pytest.raises(ValidationError, match="low 50 is above high 40"):
        talk(progress=[{"goal": "x", "low": 50, "high": 40}])


# --- masking ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("means", "sent_as", "expected"),
    [
        ("trainee:x", "[TRAINEE_35]", Outcome.PERSON_RIGHT),
        ("trainee:x", "ו[TRAINEE_9]", Outcome.PERSON_WRONG),
        ("trainee:x", "[NAME_1]", Outcome.PERSON_NEUTRAL),
        ("trainee:x", "אגם", Outcome.PERSON_LEAKED),
        ("word", "לאגם", Outcome.WORD_KEPT),
        ("word", "כ[NAME_1]", Outcome.WORD_NEUTRAL),
        ("word", "ל[TRAINEE_35]", Outcome.WORD_AS_PERSON),
    ],
)
def test_each_outcome(means, sent_as, expected):
    assert classify(Mention(word="אגם", means=means), sent_as, "[TRAINEE_35]") is expected


def test_a_word_tied_to_the_trainee_and_a_full_name_are_found_in_the_sent_text(repos, roster):
    notes = "שיחה עם אגם. היא דיברה עם איתן רוזן. בטיול לאגם היא סחבה ציוד."
    c = talk(
        notes=notes,
        mentions=[
            {"word": "אגם", "means": "trainee:אגם לנדאו"},
            {"word": "איתן", "means": "trainee:איתן רוזן"},  # a full name: two words, one token
            {"word": "לאגם", "means": "word"},
        ],
    )
    results = measure(resolve(c, repos.staff, repos.trainees, repos.goals), roster)
    assert [(r.sent_as, r.outcome) for r in results] == [
        ("[TRAINEE_35]", Outcome.PERSON_RIGHT),
        ("[TRAINEE_30]", Outcome.PERSON_RIGHT),
        ("ל[TRAINEE_35]", Outcome.WORD_AS_PERSON),  # the lake became the trainee: what this measures
    ]


def test_a_marked_word_in_a_health_sentence_is_refused(repos, roster):
    c = talk(
        notes="שיחה עם אגם. אגם הלכה לרופא בגלל כאבי ראש.",
        mentions=[{"word": "אגם", "occurrence": 2, "means": "trainee:אגם לנדאו"}],
    )
    with pytest.raises(ValueError, match="removed before sending"):
        measure(resolve(c, repos.staff, repos.trainees, repos.goals), roster)


def test_on_the_whole_talk_set_no_name_ever_leaks(repos, roster):
    """The privacy line of the measurement: whatever else happens, a person is never sent by name."""
    results = []
    for c in load_talk_set(TALK_SET).cases:
        results += measure(resolve(c, repos.staff, repos.trainees, repos.goals), roster)
    counts = summary(results)
    assert counts[Outcome.PERSON_LEAKED] == 0
    assert sum(counts.values()) == sum(len(c.mentions) for c in load_talk_set(TALK_SET).cases)


# --- the command line ---------------------------------------------------------------------------------


@pytest.fixture
def cli(repos, monkeypatch):
    import pacer.evaluation.__main__ as cli

    monkeypatch.setattr(cli, "_uow", lambda: FakeUnitOfWork(repos))
    return cli


def test_check_reports_both_sets(cli):
    report = cli.check()
    assert "library@v1" in report and "talks@v1" in report and "✗" not in report


def test_the_masking_report_has_the_table_and_every_case(cli, tmp_path):
    out = tmp_path / "masking.md"
    cli.main(["masking", "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert text.startswith('<div dir="rtl">') and "| מה קרה | כמה |" in text
    assert all(f"**{c.id}**" in text for c in load_talk_set(TALK_SET).cases)


def test_two_names_side_by_side_each_map_to_their_own_token(repos, roster):
    """Two replaced words in a row are one block for difflib: each must still find its own token."""
    c = talk(
        notes="שיחה עם אגם ושקד על התורנות.",
        mentions=[{"word": "אגם", "means": "trainee:אגם לנדאו"}, {"word": "ושקד", "means": "trainee:שקד ששון"}],
    )
    results = measure(resolve(c, repos.staff, repos.trainees, repos.goals), roster)
    assert [r.sent_as for r in results] == ["[TRAINEE_35]", "ו[TRAINEE_4]"]
