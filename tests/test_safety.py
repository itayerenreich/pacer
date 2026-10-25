"""The safety layer: every risk in the set is found, everyday words are not, and the referral is fixed."""

import time
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
        ("ה\u05beאלימות בחדר", VIOLENCE),  # a real maqaf (U+05BE) after the prefix
        ("ה-אלימות בחדר", VIOLENCE),  # an ASCII hyphen
        ("הוא אמר שהוא רוצה\u05beלמות", SELF_HARM),  # a maqaf BETWEEN the words of a phrase
        ("פוגע,בעצמו", SELF_HARM),  # a comma with no space
        ("הִתְאַבֵּד", SELF_HARM),  # niqqud
        ("הת\u200fאבד", SELF_HARM),  # an invisible bidi mark inside the word
        ("הת\u200bאבד", SELF_HARM),  # a zero-width space inside the word
    ],
)
def test_words_are_found_however_they_are_written(text, kind):
    assert check_safety(text).kinds == frozenset({kind})


@pytest.mark.parametrize(
    "text",
    ["אני אתאבד", "הוא יתאבד", "היא תתאבד", "הם יתאבדו", "נתאבד", "אני אפגע בעצמי", "הוא יפגע בעצמו"],
)
def test_every_person_and_tense_is_found(text):
    assert SELF_HARM in check_safety(text).kinds


@pytest.mark.parametrize(
    "text",
    [
        "אל תאבד תקווה",  # lose, not kill oneself: why the future forms are listed one by one
        "פגיעה בעצמאות של החניכים",  # independence, not "עצמו"
        "האישור תקף עד יום חמישי",  # valid
        "אלימלך הגיע לשיחה",  # a name that starts like "אלים"
        "אבא שלו מכהן כראש המועצה",  # starts like "מכה"
        "רבין נרצח ב-1995",  # history
        "בתורנות הוא חתך את היד בטעות",  # an accident
    ],
)
def test_near_words_are_not_found(text):
    """Each of these was a false alarm in the review (or is one step away from one)."""
    assert not check_safety(text).is_risk


def test_harm_by_a_family_member_is_also_abuse():
    """The kind decides the steps: abuse adds the reporting duty, which violence alone does not."""
    for text in ("אבא שלה הרביץ לה בשבת", "אח שלו הגדול מכה אותו", "אמא שלו מרביצה לו"):
        assert ABUSE in check_safety(text).kinds, text


def test_sexual_abuse_is_also_sexual_harm():
    assert {ABUSE, SEXUAL} <= check_safety("הוא מתעלל בה מינית").kinds


def test_long_text_is_fast():
    """No catastrophic backtracking: a 20,000-character text of near-matches takes milliseconds."""
    texts = ["פוגע " * 4000, "לא רוצה " * 2500, "ה-" * 10000, "אבא שלו " * 2500]
    started = time.perf_counter()
    for text in texts:
        check_safety(text)
    assert time.perf_counter() - started < 1.0


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


def test_abuse_and_sexual_harm_carry_the_reporting_duty_for_minors():
    for kind in (ABUSE, SEXUAL):
        actions = " ".join(s.action for s in referral_for(SafetyCheck(frozenset({kind}))).steps)
        assert "מתחת לגיל 18" in actions and "חובת דיווח" in actions
    actions = " ".join(s.action for s in referral_for(SafetyCheck(frozenset({VIOLENCE}))).steps)
    assert "חובת דיווח" not in actions  # a fight between trainees is not, by itself, a report


def test_self_harm_says_to_remove_the_means():
    actions = " ".join(s.action for s in referral_for(SafetyCheck(frozenset({SELF_HARM}))).steps)
    assert "להרחיק אמצעים" in actions
