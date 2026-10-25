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
import unicodedata
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
# Before matching, the text is cleaned (_clean): Unicode NFC, niqqud and invisible characters removed,
# and hyphens, maqaf and punctuation turned into spaces. So "רוצה־למות", "פוגע,בעצמו" and "ה-אלימות"
# are matched like plain words separated by a space.
#
# Hebrew attaches prefix letters to a word (ו ה ב כ ל מ ש: "וכשהרביץ"), so every word may start with
# up to four of them. A word starts where no letter is before it, and ends where no letter follows.
_LETTER = r"[^\W\d_]"
_START = rf"(?<!{_LETTER})[ובכלמשה]{{0,4}}"
_END = rf"(?!{_LETTER})"
_GAP = r"(?:\s+\S+){0,3}?\s+"  # whitespace, with up to three words in it: "פוגע שוב בעצמו"
_OBJ = rf"(?:את|אות(?:ו|ה|י|ם|ן|נו|ך|כם)){_END}"  # "את", "אותו", "אותה"...
# "עצמו", "בעצמה", but not "עצמאות" (independence, a core value word at an academy)
_SELF = rf"ב?עצמ(?:ו|ה|י|ם|ן|נו|ך|כם){_END}"


def _words(*words: str) -> str:
    """Whole words (each may take prefix letters)."""
    return rf"{_START}(?:{'|'.join(words)}){_END}"


def _stems(*stems: str) -> str:
    """Any word that starts with one of these (after prefix letters): every inflection."""
    return rf"{_START}(?:{'|'.join(stems)}){_LETTER}*"


def _either(*options: str) -> str:
    return "(?:" + "|".join(options) + ")"


# Verbs that harm the one who does them when followed by "עצמו": "פוגע בעצמו", "תלה את עצמו".
_SELF_VERBS = _either(
    "פוגע", "פוגעת", "פוגעים", "פוגעות", "פגע", "פגעה", "פגעו", "פגעתי", "לפגוע", "יפגע", "תפגע", "אפגע",
    "יפגעו", "תפגעי", "פגיעה", "פגיעות", "הכה", "היכה", "הכתה", "מכה", "לתלות", "תלה", "תלתה", "יתלה",
    "שורט", "שורטת", "שרט", "שרטה", "לשרוט", "מרעיב", "מרעיבה", "הרעיב", "הרעיבה", "חותך", "חותכת",
    "חתך", "חתכה", "לחתוך", "שורף", "שורפת", "שרף", "שרפה",
)
# Family members, as a subject: "אבא שלה הרביץ לה", "בן הזוג של אמא שלה נוגע בה"
_FAMILY = rf"{_START}" + _either(
    "אבא", "אמא", "אבי", "אביו", "אביה", "אמו", "אמה", "הורים", "הוריו", "הוריה", "אח", "אחות", "אחים",
    "סבא", "דוד", "אבא\\s+חורג", r"בן\s+הזוג\s+של\s+(?:אמא|אמו|אמה)",
) + _END
_FAMILY_HARM = _either(
    r"מכ(?:ה|ים)", rf"מרביצ{_LETTER}*", "מרביץ", rf"הרביצ{_LETTER}*", "הרביץ", rf"היכ{_LETTER}*", "הכה",
    rf"מתעלל{_LETTER}*", rf"התעלל{_LETTER}*", rf"משפיל{_LETTER}*", rf"נועל{_LETTER}*",
)
_TOUCHES = _either("נוגע", "נגע", "נגעו", "נוגעים", "נוגעת", "נגעה")

_PATTERNS: dict[RiskKind, list[str]] = {
    RiskKind.SELF_HARM: [
        # every tense: התאבדות, להתאבד, אתאבד, יתאבד. Listed one by one: a loose "ת?אבד" would catch
        # "אל תאבד תקווה" (do not lose hope).
        _stems("התאבד", "מתאבד", "יתאבד", "תתאבד", "אתאבד", "נתאבד", "אובדנ"),
        rf"{_START}{_SELF_VERBS}{_GAP}(?:את\s+)?{_SELF}",  # "פוגעת שוב ושוב בעצמה", "תלה את עצמו"
        rf"{_START}פגיע(?:ה|ות)\s+עצמי(?:ת|ות)?{_END}",
        rf"{_START}{_SELF_VERBS}{_GAP}ב?(?:ה)?גוף\s+של",  # "פוגעת בגוף שלה"
        _words("חתכים"),
        rf"{_START}נחתכ{_LETTER}*{_GAP}בכוונה",
        # "חתך את היד", but not "חתך את היד בטעות" (kitchen duty)
        rf"{_START}(?:חותך|חותכת|חתך|חתכה|לחתוך)\s+(?:את\s+)?(?:ה?ידיים|ה?יד|ידיו|ידיה|ה?זרוע)"
        rf"{_END}(?!(?:\s+\S+){{0,3}}?\s+בטעות)",
        # a wish to die, or a question about how. Never "למות" alone ("מוכנה למות על הכבוד שלה", N10),
        # and not the idioms "למות מצחוק / מבושה / מעייפות"
        rf"{_START}(?:רוצה|רוצים|רצה|רצתה|מעדיף|מעדיפה|עדיף|מתכנן|מתכננת|חושב|חושבת|בא|כואב|איך(?!\s+לא))"
        rf"{_GAP}למות{_END}(?!\s+מ(?:צחוק|בושה|עייפות|רעב|קנאה|שעמום|פחד))",
        rf"לא\s+(?:רוצה|רוצים|רצה|רצתה){_GAP}(?:לחיות|להתעורר){_END}",
        r"כדי\s+לא\s+להתעורר|לא\s+לקום\s+יותר",
        r"אין\s+(?:\S+\s+)?טעם\s+(?:\S+\s+)?לחיות|נמאס\s+(?:\S+\s+)?(?:לחיות|מהחיים)",
        r"אין\s+(?:לו|לה|לי|להם)\s+(?:בשביל\s+)?(?:מה|למה)\s+לחיות",
        r"לשים\s+קץ|לסיים\s+את\s+(?:ה)?חיים",
        rf"{_START}(?:לגמור|לגמר|גומר|גומרת)\s+(?:עם\s+)?(?:{_SELF}|הכל{_END}|הכול{_END}|החיים{_END})",
        rf"{_START}(?:ישים|לשים|שם|שמה|תשים|אשים)\s+יד\s+על\s+{_SELF}",
        rf"{_START}(?:לקחת|לקח|לקחה|ניסה|ניסתה|ניסו){_GAP}את\s+(?:ה)?חיים\s+של",
        rf"עדיף\s+(?:ש{_LETTER}*\s+)?לא\s+(?:יהיה|תהיה|אהיה|נהיה)\s+(?:פה|כאן|בחיים|קיים|קיימת){_END}",
        rf"טוב\s+בלעדי(?:ו|ה|יי|י){_END}",
        r"מכתב\s+(?:ה)?פרידה",  # also an academy's end-of-year tradition: a known false alarm, kept
        rf"{_START}(?:נפרד|נפרדת|נפרדה)\s+מכולם",
        # pills: "בלע כדורים", "לקח המון כדורים". "כדורים" alone is also balls: the amount is required
        rf"{_START}(?:בלע|בלעה|לבלוע|יבלע|תבלע){_GAP}(?:את\s+)?(?:הכול|הכל|ה?כדורים|ה?תרופות|את\s+כל){_END}",
        rf"{_START}(?:לקח|לקחה|לקחת){_GAP}(?:הרבה|המון|חופן|קופסת|קופסה\s+של|את\s+כל\s+ה)\s*כדורים",
        r"כמה\s+כדורים\s+(?:צריך|צריכים|צריכה)",
        rf"{_START}(?:לקפוץ|יקפוץ|תקפוץ|אקפוץ|קפץ|קפצה|קפיצה){_GAP}מ(?:ה)?(?:גג|גשר|קומה|חלון)",
    ],
    RiskKind.VIOLENCE: [
        _words(
            "אלימות", "אלים", "אלימה", "אלימים", "מכות", "תקיפה", "תקיפות", "דקירה", "הותקף", "הותקפה",
            "הותקפו", "אגרוף", "אגרופים", "סטירה", "סטירות", "מאוים", "מאוימת", "מאוימים",
        ),
        _words("איום", "איומים", "איים", "איימה", "איימו", "מאיים", "מאיימת", "מאיימים"),
        _stems("הרביצ", "הרביץ", "מרביצ", "מרביץ", "ירביץ", "תרביץ", "ארביץ", "נרביץ"),  # להרביץ, הרביצו
        # intent to kill. Not "נרצח" or "רצח": the lesson about Rabin's murder (N04)
        _words("להרוג", "אהרוג", "יהרוג", "תהרוג", "נהרוג", "לרצוח", "ארצח", "ירצח", "תרצח"),
        _stems("דקר", "ידקור", "תדקור", "אדקור", "נדקור", "לדקור"),
        rf"{_START}(?:תקף|תקפה|תקפו|תוקף|תוקפת|תוקפים){_GAP}{_OBJ}",  # not "האישור תקף"
        rf"{_START}(?:שלף|שלפה|שלפו){_GAP}(?:סכין|נשק|אקדח)",
        rf"{_START}(?:מכים|מכה|היכה|הכה|היכתה|הכתה|היכו|הכו|להכות|יכה|תכה|אכה){_GAP}(?:{_OBJ}|חניך|חניכה|ילד)",
        rf"{_START}(?:ישבור|אשבור|תשבור|נשבור)\s+(?:לו|לה|לך|להם)\s+את",
        rf"{_START}(?:חנק|חנקה|חנקו|חונק|לחנוק|בעט|בעטה|בעטו|בועט|בועטים|התנפל|התנפלה|התנפלו)"
        rf"\s+(?:ב(?:ו|ה|הם){_END}|על{_END}|{_OBJ}|חניך|חניכה)",
        rf"{_START}(?:דחף|דחפה|דחפו)\s+{_OBJ}\s+(?:\S+\s+)?(?:לקיר|לרצפה|חזק)",
        rf"{_START}(?:יירה|ירה|לירות|אירה)\s+ב(?:כולם|ו|ה|הם){_END}",
    ],
    RiskKind.ABUSE: [
        _stems("התעלל", "מתעלל"),  # התעללות, התעללו, מתעללים
        _words("הזנחה", "מוזנח", "מוזנחת", "מזניחים"),
        r"אלימות\s+(?:ב|של\s+ה|מצד\s+ה)?(?:משפחה|בית|הורים|אבא|אמא)",
        # hit at home, but not "at school" ("בבית הספר") or "went home" ("הביתה")
        rf"{_START}(?:מכות|מכים|מרביצים|היכו|מכה|מרביץ|הרביץ|הרביצו|הכה|היכה){_END}{_GAP}"
        rf"(?:{_OBJ}\s+)?(?:ב|ב?ה)?(?:בית|משפחה){_END}(?!\s+ה?ספר)",
        rf"{_FAMILY}{_GAP}{_FAMILY_HARM}{_END}",
        rf"{_START}(?:מפחד|מפחדת|מפחדים)\s+לחזור\s+(?:ה)?בית",
        rf"{_START}(?:זרק|זרקה|זרקו){_GAP}(?:{_OBJ}\s+)?מהבית",
        r"סימנים\s+כחולים",
        rf"{_START}(?:נועל|נועלת|נועלים|נעל|נעלה|נעלו)\s+{_OBJ}\s+(?:\S+\s+)?(?:בחדר|בבית|בארון)",
        rf"לא\s+(?:דואג|דואגת|דואגים|נותן|נותנת|נותנים)\s+(?:לו|לה)\s+(?:\S+\s+)?(?:ל)?(?:אוכל|לאכול){_END}",
        r"אין\s+אוכל\s+בבית",
        rf"{_START}(?:ישן|ישנה|ישנים)\s+ברחוב",
    ],
    RiskKind.SEXUAL_HARM: [
        _words(
            "אונס", "אנס", "אנסה", "אנסו", "לאנוס", "נאנס", "נאנסה", "נאנסו", "הטרדה", "הטרדות",
            "עירום", "עירומה", "עירומות",
        ),
        rf"{_START}(?:פגיעה|פגיעות|פגע|פגעה|פוגע|תקיפה|תקף|תקפה|תקפו|הותקף|הותקפה|הותקפו|ניצול|ניצל|ניצלה|ניצלו"
        rf"|הטריד|הטרידה|מטריד|מטרידה|הטרדה|מגע|התעלל|התעללה|התעללו|מתעלל|מתעללת|הודעות|תוכן|תמונות)"
        rf"{_GAP}מיני{_LETTER}*",
        r"מגע\s+(?:לא|שלא)\s+(?:רצוי|הולם|מתאים|רצה|רצתה)",
        rf"{_START}(?:הכריח|הכריחה|הכריחו|מכריח|מכריחה|מכריחים|לחץ\s+עלי(?:ה|ו)){_GAP}"
        r"(?:\S+\s+)?(?:לקיים|לשכב|יחסים|לעשות\s+דברים)",
        rf"{_START}(?:צילם|צילמה|צילמו|מצלם|הפיץ|הפיצה|הפיצו){_GAP}(?:\S+\s+)?(?:במקלחת|בשירותים|עירומ|תמונות\s+אינטימיות)",
        rf"{_START}{_TOUCHES}\s+ב(?:ה|ו|י){_END}(?:\s+\S+){{0,3}}?\s*(?:בלי|ללא|נגד|בכוח|בלילה|במקומות|באיברים)",
        rf"{_FAMILY}{_GAP}{_TOUCHES}{_END}",
        rf"{_START}(?:נכנס|נכנסה|נכנסו)\s+(?:לה|לו)\s+(?:ל|ה)?מיטה",
    ],
}

# Niqqud and invisible characters (bidi marks, zero-width spaces) would split a word in two:
# "הִתְאַבֵּד" and "הת<zero-width>אבד" must be found like "התאבד". The maqaf (U+05BE) sits inside the
# niqqud range but joins two WORDS, so it is left out here and turned into a space below.
_INVISIBLE = re.compile("[\u0591-\u05bd\u05bf-\u05c7\u00ad\u061c\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]")
_SPACE = re.compile(r"[-\u05be\u2010-\u2015\u2212,;:!?.()\[\]\"'/\\\u05f4\u05f3\u201c\u201d\u2018\u2019`\u00b4\u2026]+")
_COMPILED = {kind: re.compile("|".join(patterns)) for kind, patterns in _PATTERNS.items()}


def _clean(text: str) -> str:
    return _SPACE.sub(" ", _INVISIBLE.sub("", unicodedata.normalize("NFC", text)))


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
        clean = _clean(text)
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
# For a minor the law decides, not the manager: every adult must report an offense against a minor by
# the one responsible for them or by a family member, and education staff more strictly; an offense
# inside the academy must be reported whoever committed it (kolzchut.org.il, checked 2026-10-25).
# Most trainees are 18, some are not, so the step says both. Not legal advice: the academy checks it.
_REPORTING = ReferralStep(
    "חניך מתחת לגיל 18: יש חובת דיווח בחוק, ולאנשי חינוך היא מחמירה. מדווחים לרווחה (המחלקה לשירותים "
    "חברתיים) או למשטרה. לעדכן את המנהל ולוודא שהדיווח נעשה. מעל 18: המנהל מחליט עם גורם מקצועי.",
    "רווחה · משטרה 100",
)
_STEPS_BY_KIND: dict[RiskKind, tuple[ReferralStep, ...]] = {
    RiskKind.SELF_HARM: (
        ReferralStep(
            "לשאול את החניך ישירות ובעדינות אם הוא חושב לפגוע בעצמו. שאלה ישירה לא מעודדת פגיעה.",
        ),
        ReferralStep("להרחיק אמצעים זמינים (תרופות, חפצים חדים, נשק), בלי להשאיר את החניך לבד."),
        ReferralStep("עזרה ראשונה נפשית, 24 שעות, גם בשבילך כמדריך.", "ער\"ן 1201"),
    ),
    RiskKind.VIOLENCE: (
        ReferralStep("להרחיק את הצדדים זה מזה ולוודא שאף אחד לא בסכנה, לפני כל שיחה."),
    ),
    RiskKind.ABUSE: (
        _REPORTING,
    ),
    RiskKind.SEXUAL_HARM: (
        ReferralStep(
            "קו סיוע לנפגעות ולנפגעי תקיפה מינית, 24 שעות, אנונימי.",
            "1202 (נשים) · 1203 (גברים)",
        ),
        _REPORTING,
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
