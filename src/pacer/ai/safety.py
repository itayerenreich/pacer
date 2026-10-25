"""Risk situations: found in code, BEFORE any model, and answered with a referral, never with advice.

The counselor coach (stage 5) gives educational advice. Some questions are not educational questions:
a trainee who talks about ending their life, violence, abuse, sexual harm. For those the right answer
is not a model's idea, it is the academy's procedure: who to tell, now, and who to call.

    check_safety(text)   which risk kinds the text shows (empty: no risk found)
    referral_for(check)  the fixed answer: what to do, in order, with phone numbers

Why code and not a model (decision 2026-10-25):
* The safety path must not depend on a model being up, being right, or seeing the text. A local
  model is slow and weak on this laptop (stage 4 measured it); a cloud model would mean sending the
  most sensitive text there is to the cloud. A phrase list is instant, deterministic, and testable.
* Every phrasing it catches is a line in evals/safety-v1.json, and a test demands ALL of them.

Over-matching is the safe direction, as in sensitive.py, but the cost is different: a false alarm
here means the counselor gets a referral instead of advice and has to rephrase. So words that are
everyday in an academy are deliberately NOT on the list, only the phrases that make them a risk:
"נשק" and "סכין" (weapons training, kitchen duty), "רצח" (a lesson about Rabin's murder), "מטריד"
(worrying), "נפצע" (an injury at training). Each such choice has a near-miss case in the set.

Not on the list on purpose: social exclusion and bullying (חרם). It hurts, and it is exactly what a
counselor needs educational advice about; the coach answers it, with the manager in "who to involve".

What this module does NOT do: diagnose, or decide that there is no risk. Finding nothing means only
that no phrase matched; the counselor's judgment and the academy's procedure always come first.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class RiskKind(StrEnum):
    SELF_HARM = "self_harm"  # suicide, self-harm, a wish to die
    VIOLENCE = "violence"  # violence or a threat, by the trainee or toward them
    ABUSE = "abuse"  # abuse or neglect, at home or by others
    SEXUAL_HARM = "sexual_harm"  # sexual assault or harassment

    @property
    def label(self) -> str:
        return _LABELS[self]


_LABELS = {
    RiskKind.SELF_HARM: "פגיעה עצמית",
    RiskKind.VIOLENCE: "אלימות או איום",
    RiskKind.ABUSE: "התעללות או הזנחה",
    RiskKind.SEXUAL_HARM: "פגיעה מינית",
}

# --- the patterns --------------------------------------------------------------------------------
# Hebrew attaches prefix letters to a word (ו ה ב כ ל מ ש: "וכשהרביץ"), so every word may start
# with up to four of them, and may be written with a maqaf after them ("ה-אלימות").
# A word starts where no letter is before it, so "תקיפה" is not found inside "שתקיפה" by accident...
# except as ש + תקיפה, which is exactly the prefix rule.
_LETTER = r"[^\W\d_]"
_START = rf"(?<!{_LETTER})(?:[ובכלמשה]{{0,4}}[-־]?)"
_END = rf"(?!{_LETTER})"
_GAP = r"(?:\s+\S+){0,3}?\s+"  # whitespace, with up to three words in it: "פוגע שוב בעצמו"


def _words(*words: str) -> str:
    """Whole words (each may take prefix letters)."""
    return rf"{_START}(?:{'|'.join(words)}){_END}"


def _stems(*stems: str) -> str:
    """Any word that starts with one of these (after prefix letters): every inflection."""
    return rf"{_START}(?:{'|'.join(stems)}){_LETTER}*"


_PATTERNS: dict[RiskKind, list[str]] = {
    RiskKind.SELF_HARM: [
        _stems("התאבד", "מתאבד", "אובדנ"),  # התאבדות, להתאבד, מחשבות אובדניות
        # "פוגע בעצמו", "לפגוע שוב בעצמה", "פגיעה עצמית"
        rf"{_START}(?:פוגע|פוגעת|פוגעים|פגע|פגעה|פגעו|לפגוע|יפגע|תפגע|פגיעה|פגיעות){_GAP}ב?עצמ{_LETTER}*",
        rf"{_START}פגיע(?:ה|ות){_GAP}עצמי{_LETTER}*",
        _words("חתכים"),
        rf"{_START}(?:חותך|חותכת|חתך|חתכה|לחתוך){_GAP}(?:את\s+)?(?:עצמ{_LETTER}*|ה?ידיים|ה?יד|ידיו|ידיה)",
        # a wish to die, or a question about how: "רוצה למות", "בא לו למות", "אם כואב למות". Never "למות"
        # alone: "מוכנה למות על הכבוד שלה" is stubbornness (near miss N10)
        rf"{_START}(?:רוצה|רוצים|רצה|רצתה|מעדיף|מעדיפה|עדיף|מתכנן|מתכננת|חושב|חושבת|בא|כואב|איך){_GAP}למות{_END}",
        rf"לא\s+(?:רוצה|רוצים|רצה|רצתה|רוצה\s+להמשיך|רוצה\s+יותר){_GAP}לחיות{_END}",
        r"אין\s+(?:\S+\s+)?טעם\s+(?:\S+\s+)?לחיות|נמאס\s+(?:\S+\s+)?(?:לחיות|מהחיים)",
        r"לשים\s+קץ|לסיים\s+את\s+(?:ה)?חיים",
        r"מכתב\s+(?:ה)?פרידה",
        rf"{_START}(?:בלע|בלעה|לבלוע|לקח|לקחה){_GAP}(?:הרבה|המון|חופן|קופסת|קופסה\s+של|את\s+כל\s+ה)?\s*כדורים",
        rf"{_START}(?:לבלוע|בלע|בלעה){_GAP}(?:את\s+)?(?:הכול|הכל|הכדורים)",  # "כדורים" alone is also balls
        rf"{_START}(?:לקפוץ|יקפוץ|תקפוץ|אקפוץ|קפץ|קפצה|קפיצה){_GAP}מ(?:ה)?(?:גג|גשר|קומה|חלון)",
    ],
    RiskKind.VIOLENCE: [
        _words("אלימות", "אלים", "אלימה", "אלימים", "מכות", "תקיפה", "תקיפות", "דקירה", "הותקף", "הותקפה", "הותקפו"),
        _words("איום", "איומים", "איים", "איימה", "איימו", "מאיים", "מאיימת", "מאיימים"),
        _stems("הרביצ", "הרביץ", "מרביצ", "מרביץ", "ירביץ", "תרביץ", "ארביץ", "נרביץ"),  # להרביץ, הרביצו
        _words("להרוג", "אהרוג", "יהרוג", "תהרוג", "נהרוג", "לרצוח", "ארצח", "ירצח", "תרצח", "נרצח"),
        _stems("דקר"),  # דקר, דקרה, דקרו
        rf"{_START}(?:תקף|תקפה|תקפו|תוקף|תוקפת|תוקפים){_GAP}(?:את|אות){_LETTER}*",  # not "האישור תקף"
        rf"{_START}(?:שלף|שלפה|שלפו){_GAP}(?:סכין|נשק|אקדח)",
        rf"{_START}(?:מכים|מכה|היכה|היכתה|היכו|להכות){_GAP}(?:אות|את){_LETTER}*",
    ],
    RiskKind.ABUSE: [
        _stems("התעלל", "מתעלל"),  # התעללות, התעללו, מתעללים
        _words("הזנחה", "מוזנח", "מוזנחת", "מזניחים"),
        r"אלימות\s+(?:ב|של\s+ה|מצד\s+ה)?(?:משפחה|בית|הורים|אבא|אמא)",
        rf"(?:מכות|מכים|מרביצים|היכו|מכה|מרביץ){_GAP}(?:אות{_LETTER}*\s+)?(?:ב|ה)?(?:בית|משפחה)",
        r"(?:אבא|אמא|הורים|אח|אחות)\s+(?:\S+\s+)?(?:שלו|שלה)?\s*(?:מכים|מכה|מרביץ|מרביצה|מרביצים)",
    ],
    RiskKind.SEXUAL_HARM: [
        _words("אונס", "אנס", "אנסה", "אנסו", "נאנס", "נאנסה", "נאנסו", "הטרדה", "הטרדות"),
        rf"{_START}(?:פגיעה|פגיעות|פגע|פגעה|פוגע|תקיפה|תקף|הותקף|הותקפה|הותקפו|ניצול|ניצל|הטריד|הטרידה|מטריד|מטרידה|הטרדה|מגע){_GAP}מיני{_LETTER}*",
        r"מגע\s+(?:לא|שלא)\s+(?:רצוי|הולם|מתאים|רצה|רצתה)",
    ],
}

# Niqqud and invisible characters (bidi marks, zero-width spaces) would split a word in two:
# "הִתְאַבֵּד" and "הת​אבד" must be found like "התאבד". The same set the anonymizer removes.
_INVISIBLE = re.compile("[֑-ׇ­؜​-‏‪-‮⁠-⁩﻿]")
_COMPILED = {kind: re.compile("|".join(patterns), re.IGNORECASE) for kind, patterns in _PATTERNS.items()}


@dataclass(frozen=True)
class SafetyCheck:
    kinds: frozenset[RiskKind]

    @property
    def is_risk(self) -> bool:
        return bool(self.kinds)


def check_safety(*texts: str) -> SafetyCheck:
    """Which risk kinds these texts show. Checks each text whole, and all of them: a question, the
    situation the counselor described, anything the coach is about to work on."""
    found: set[RiskKind] = set()
    for text in texts:
        clean = _INVISIBLE.sub("", text)
        found.update(kind for kind, pattern in _COMPILED.items() if pattern.search(clean))
    return SafetyCheck(frozenset(found))


# --- the referral --------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferralStep:
    action: str  # what to do, in the counselor's words
    contact: str | None = None  # a phone number or a person, when there is one


@dataclass(frozen=True)
class Referral:
    kinds: tuple[RiskKind, ...]
    message: str
    steps: tuple[ReferralStep, ...]


# A GENERAL default, written for the demo academy. An academy's own procedure replaces it: its
# contacts, its order, its words. Numbers checked on 2026-10-25 (ער"ן; the national rape crisis lines,
# kolzchut.org.il). Keep every step short: this is read by a counselor in a hard moment.
_FIRST_STEPS = (
    ReferralStep(
        "אם יש סכנה מיידית לחיים: לא להשאיר את החניך לבד, ולהתקשר מיד למד\"א או למשטרה.",
        "מד\"א 101 · משטרה 100",
    ),
    ReferralStep("לעדכן עכשיו את מנהל המכינה. זה לא מצב שמטפלים בו לבד.", "מנהל המכינה"),
    ReferralStep(
        "לכתוב בקצרה מה נאמר ומתי, במילים של מי שאמר, ולמסור למנהל. לא לחקור, ולא להבטיח לשמור בסוד."
    ),
)
_STEPS_BY_KIND: dict[RiskKind, tuple[ReferralStep, ...]] = {
    RiskKind.SELF_HARM: (
        ReferralStep(
            "לשאול את החניך ישירות ובעדינות אם הוא חושב לפגוע בעצמו. שאלה ישירה לא מעודדת פגיעה.",
        ),
        ReferralStep("עזרה ראשונה נפשית, 24 שעות, גם בשבילך כמדריך.", "ער\"ן 1201"),
    ),
    RiskKind.VIOLENCE: (
        ReferralStep("להרחיק את הצדדים זה מזה ולוודא שאף אחד לא בסכנה, לפני כל שיחה."),
    ),
    RiskKind.ABUSE: (
        ReferralStep("המנהל יחליט, יחד עם גורם מקצועי, אם ואיך לדווח ולמי."),
    ),
    RiskKind.SEXUAL_HARM: (
        ReferralStep(
            "קו סיוע לנפגעות ולנפגעי תקיפה מינית, 24 שעות, אנונימי.",
            "1202 (נשים) · 1203 (גברים)",
        ),
        ReferralStep("המנהל יחליט, יחד עם גורם מקצועי, אם ואיך לדווח ולמי."),
    ),
}


def referral_for(check: SafetyCheck) -> Referral:
    """The fixed answer to a risk situation: the general steps, then the steps for each kind found,
    each step once. No model is involved, so this answer is the same every time and never fails."""
    if not check.is_risk:
        raise ValueError("referral_for() needs a check that found a risk")
    kinds = tuple(kind for kind in RiskKind if kind in check.kinds)  # a fixed order, not set order
    labels = ", ".join(kind.label for kind in kinds)
    message = (
        f"מה שתיארת עלול להיות מצב סיכון ({labels}). במצבים כאלה Pacer לא נותן ייעוץ, "
        "כי הם דורשים אדם ולא מערכת. אלה הצעדים לפי הנוהל:"
    )
    return Referral(kinds=kinds, message=message, steps=_unique(_FIRST_STEPS, *(_STEPS_BY_KIND[k] for k in kinds)))


def _unique(*groups: Iterable[ReferralStep]) -> tuple[ReferralStep, ...]:
    steps: dict[ReferralStep, None] = {}
    for group in groups:
        steps.update(dict.fromkeys(group))
    return tuple(steps)
