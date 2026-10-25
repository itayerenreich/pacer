"""The safety layer: every risk in the set is found, everyday words are not, and the referral is fixed."""

from pathlib import Path

import pytest

from pacer.ai.safety import Referral, RiskKind, SafetyCheck, check_safety, referral_for
from pacer.evaluation.safety_set import evaluate, load_safety_set

SAFETY_SET = Path(__file__).parent.parent / "evals" / "safety-v1.json"
SELF_HARM, VIOLENCE, ABUSE, SEXUAL = RiskKind.SELF_HARM, RiskKind.VIOLENCE, RiskKind.ABUSE, RiskKind.SEXUAL_HARM


# --- the evaluation set: the real bar -----------------------------------------------------------


def test_every_risk_case_in_the_set_is_found():
    """Recall must be 100%: a missed case means advice where the procedure was needed."""
    result = evaluate(load_safety_set(SAFETY_SET))
    assert result.missed == []
    assert result.recall == 1.0


def test_no_near_miss_in_the_set_becomes_a_referral():
    """Each near miss is the reason a word is kept off the list. Adding it back fails here."""
    result = evaluate(load_safety_set(SAFETY_SET))
    assert result.false_alarms == []


def test_the_set_covers_every_kind():
    kinds = {k for case in load_safety_set(SAFETY_SET).cases for k in case.expect}
    assert kinds == set(RiskKind)


# --- how words are found -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("וכשהתאבדותו של חבר עלתה בשיחה", SELF_HARM),  # prefix letters: ו כ ש
        ("ה-אלימות בחדר", VIOLENCE),  # a maqaf after the prefix
        ("הִתְאַבֵּד", SELF_HARM),  # niqqud
        ("הת‏אבד", SELF_HARM),  # an invisible bidi mark inside the word
        ("להרביץ", VIOLENCE),
        ("ATTACK", None),  # nothing in English is on the list
    ],
)
def test_words_are_found_however_they_are_written(text, kind):
    assert check_safety(text).kinds == (frozenset({kind}) if kind else frozenset())


@pytest.mark.parametrize(
    "text",
    [
        "שתקיפות",  # ש + תקיפות: a prefix, found (over-matching is the safe direction)
        "מאיימים",
    ],
)
def test_prefixed_forms_are_found(text):
    assert check_safety(text).is_risk


@pytest.mark.parametrize(
    "text",
    [
        "האישור תקף עד יום חמישי",  # valid
        "חיתוך ירקות בתורנות",
        "אלימלך הגיע לשיחה",  # a name that starts like "אלים"
        "פעילות על מיתולוגיה",
    ],
)
def test_words_inside_other_words_are_not_found(text):
    assert not check_safety(text).is_risk


def test_a_long_gap_does_not_join_unrelated_words():
    """'פוגע' and 'בעצמו' far apart in a long sentence are not self-harm."""
    text = "הוא פוגע לפעמים באווירה של הקבוצה כשהוא מאחר, אבל מתקן את זה בעצמו"
    assert not check_safety(text).is_risk


def test_several_texts_are_checked_together():
    check = check_safety("איך לעזור לו עם המטרות?", "הוא אמר שהוא רוצה למות")
    assert check.kinds == {SELF_HARM}


def test_several_kinds_at_once():
    assert check_safety("חניכה סיפרה על אלימות במשפחה ועל פגיעה מינית").kinds == {VIOLENCE, ABUSE, SEXUAL}


# --- the referral --------------------------------------------------------------------------------


def test_the_referral_starts_with_immediate_danger_and_the_manager():
    referral = referral_for(SafetyCheck(frozenset({VIOLENCE})))
    assert "101" in (referral.steps[0].contact or "")
    assert referral.steps[1].contact == "מנהל המכינה"


def test_each_kind_adds_its_own_steps():
    contacts = [s.contact for s in referral_for(SafetyCheck(frozenset({SELF_HARM}))).steps]
    assert any(c and "1201" in c for c in contacts)
    contacts = [s.contact for s in referral_for(SafetyCheck(frozenset({SEXUAL}))).steps]
    assert any(c and "1202" in c and "1203" in c for c in contacts)


def test_a_step_shared_by_two_kinds_appears_once():
    referral = referral_for(SafetyCheck(frozenset({ABUSE, SEXUAL})))
    assert len(referral.steps) == len(set(referral.steps))


def test_the_referral_is_the_same_every_time_whatever_the_set_order():
    a = referral_for(SafetyCheck(frozenset({SEXUAL, SELF_HARM})))
    b = referral_for(SafetyCheck(frozenset({SELF_HARM, SEXUAL})))
    assert a == b
    assert a.kinds == (SELF_HARM, SEXUAL)  # the order of RiskKind, not of the set


def test_the_message_names_the_kinds_in_hebrew():
    referral: Referral = referral_for(SafetyCheck(frozenset({SELF_HARM})))
    assert "פגיעה עצמית" in referral.message
    assert "self_harm" not in referral.message


def test_no_referral_without_a_risk():
    with pytest.raises(ValueError):
        referral_for(SafetyCheck(frozenset()))
