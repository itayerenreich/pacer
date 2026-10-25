"""The safety gate: phrases first and final, the model only adds, nothing personal reaches it, and it fails closed."""

import json

import pytest
from fake_llm import FakeLLMClient, fake_router

from pacer.ai import LLMUnavailableError
from pacer.ai.anonymization import Anonymizer, Person, PersonKind, Roster
from pacer.ai.risk_screen import FoundBy, SafetyGate, ScreenAnswer
from pacer.ai.routing import ROUTES, AITask, Sensitivity
from pacer.ai.safety import REFERRAL, RiskKind
from pacer.ai.sensitive import HEALTH_MASK
from pacer.ai.structured import StructuredOutputError

YOAV = Person(PersonKind.TRAINEE, 7, "יואב", "לוי")
ANONYMIZER = Anonymizer(Roster([YOAV, Person(PersonKind.STAFF, 2, "נועה", "פרץ")]))


def screen(risk: bool, kinds: list[str] = (), reason: str = "a signal") -> str:  # type: ignore[assignment]
    return json.dumps({"risk": risk, "kinds": list(kinds), "reason": reason})


def gate_with(*answers: str) -> tuple[SafetyGate, FakeLLMClient]:
    cloud = FakeLLMClient(answers=list(answers), runs_locally=False)
    return SafetyGate(fake_router(cloud)), cloud


# --- the order: phrases first, model second ------------------------------------------------------


def test_a_phrase_match_is_final_and_no_model_is_asked():
    gate, cloud = gate_with(screen(False))
    decision = gate.assess("יואב אמר שהוא רוצה למות", ANONYMIZER)
    assert decision.is_risk and decision.found_by is FoundBy.PHRASES
    assert cloud.calls == []  # the most sensitive text never left the computer


def test_the_model_catches_a_scene_the_phrases_miss():
    gate, cloud = gate_with(screen(True, ["self_harm"], "stood somewhere high at night"))
    decision = gate.assess("חבר שלו סיפר שיואב עמד על הגג בלילה וחשב על דברים", ANONYMIZER)
    assert decision.is_risk and decision.found_by is FoundBy.MODEL
    assert decision.referral is REFERRAL and decision.kinds == {RiskKind.SELF_HARM}
    assert len(cloud.calls) == 1


def test_a_no_from_the_model_gives_advice_its_way():
    gate, _ = gate_with(screen(False))
    decision = gate.assess("איך לעזור ליואב לדבר יותר בשיחות הקבוצה?", ANONYMIZER)
    assert not decision.is_risk and decision.referral is None


def test_the_model_answer_gets_the_same_procedure_as_the_phrases():
    gate, _ = gate_with(screen(True, ["abuse", "sexual_harm"]))
    decision = gate.assess("משהו קרה לו בבית בשבת והוא לא מוכן לספר", ANONYMIZER)
    assert decision.referral is REFERRAL


# --- what reaches the model ----------------------------------------------------------------------


def test_names_never_reach_the_model():
    gate, cloud = gate_with(screen(False))
    gate.assess("נועה שמה לב שיואב לוי שותק כל השבוע", ANONYMIZER)
    sent = cloud.calls[0].user_message
    assert "יואב" not in sent and "נועה" not in sent
    assert "[TRAINEE_7]" in sent and "[STAFF_2]" in sent


def test_health_words_never_reach_the_model_but_the_rest_of_their_sentence_does():
    """The screen masks health WORDS (not sentences): the scene must still be screened."""
    gate, cloud = gate_with(screen(False))
    gate.assess("הפסיכולוג סיפר שיואב עמד על הגג בלילה. הוא שקט בשיחות הקבוצה.", ANONYMIZER)
    sent = cloud.calls[0].user_message
    assert "פסיכולוג" not in sent and "יואב" not in sent
    assert "עמד על הגג בלילה" in sent and HEALTH_MASK in sent


def test_a_risk_scene_in_a_sentence_about_health_is_screened():
    """The review's case: the old gate dropped the whole sentence, and the counselor got advice."""
    gate, _ = gate_with(screen(True, ["self_harm"]))
    decision = gate.assess("הפסיכולוג סיפר שיואב עמד על הגג בלילה. איך לעזור לו להשתתף בקבוצה?", ANONYMIZER)
    assert decision.found_by is FoundBy.MODEL


def test_nothing_but_health_words_means_no_call():
    gate, cloud = gate_with(screen(True, ["self_harm"]))
    decision = gate.assess("ריטלין.", ANONYMIZER)
    assert cloud.calls == [] and not decision.is_risk


def test_the_prompt_is_the_pinned_version():
    gate, cloud = gate_with(screen(False))
    gate.assess("איך לחזק את יואב?", ANONYMIZER)
    assert cloud.calls[0].prompt.ref == "risk_screen@v1"


def test_the_route_is_anonymized_data_only():
    assert ROUTES[AITask.SCREEN_RISK].sensitivity is Sensitivity.ANONYMIZED


# --- failing closed ------------------------------------------------------------------------------


def test_a_model_that_is_down_stops_everything():
    """No silent pass: an unscreened question must not get advice. The API turns this into 503."""

    def down(_message: str) -> str:
        raise LLMUnavailableError("down")

    gate, _ = gate_with()
    gate._router = fake_router(FakeLLMClient(answers=down, runs_locally=False))
    with pytest.raises(LLMUnavailableError):
        gate.assess("איך לחזק את יואב?", ANONYMIZER)


def test_a_contradictory_answer_is_corrected_once_then_refused():
    """risk=true with no kinds is not a usable answer: one correction, then StructuredOutputError."""
    bad = screen(True, [])
    gate, cloud = gate_with(bad, bad)
    with pytest.raises(StructuredOutputError):
        gate.assess("איך לחזק את יואב?", ANONYMIZER)
    assert len(cloud.calls) == 2


def test_a_corrected_answer_is_used():
    gate, _ = gate_with(screen(True, []), screen(True, ["violence"]))
    decision = gate.assess("שמעתי שמתכננים לעשות ליואב משהו בלילה", ANONYMIZER)
    assert decision.found_by is FoundBy.MODEL


@pytest.mark.parametrize(("risk", "kinds"), [(True, []), (False, ["violence"])])
def test_risk_and_kinds_must_agree(risk, kinds):
    with pytest.raises(ValueError):
        ScreenAnswer(risk=risk, kinds=kinds, reason="x")


# --- the measurement -----------------------------------------------------------------------------


def test_the_gate_evaluation_counts_each_layer_and_the_false_alarms():
    from pathlib import Path

    from pacer.evaluation.safety_set import evaluate_gate, load_safety_set, render_gate

    held_out = load_safety_set(Path(__file__).parent.parent / "evals" / "safety-heldout-v1.json")
    always = FakeLLMClient(answers=[screen(True, ["self_harm"])], runs_locally=False)
    gate = SafetyGate(fake_router(always))
    result = evaluate_gate(held_out, lambda text: gate.assess(text, ANONYMIZER))
    risk = result.risk_cases()
    assert len(risk) == 60 and all(result.caught(c) for c in risk)  # this fake flags everything
    # the phrase list alone: 33 get a referral, 30 of them with the expected kind (the 30/60 measured)
    assert sum(c.found_by == "phrases" for c in risk) == 33
    report = render_gate(result)
    assert "**60/60**" in report and "התראות שווא" in report


def test_the_free_safety_report_needs_no_model_and_no_database(capsys):
    from pacer.evaluation.__main__ import main

    main(["safety"])
    assert "30/60" in capsys.readouterr().out
