"""Health and diagnoses: found in code, and kept away from the cloud and from long-term memory.

Two uses, one list:
* remove_sensitive_sentences() drops such sentences from a counselor's notes and from a meeting
  transcript BEFORE anything is sent (decision 2026-10-18: data minimization). What the model
  never receives, it can never repeat, summarize or propose to remember.
* is_sensitive() also checks the model's answer, so a note or a progress reason about health is
  never kept (the prompt says so; this is the second line of defense, in code).

Over-matching is the safe direction: a sentence removed by mistake costs a less complete summary
(the person still has their own text); a sensitive one that gets through is a leak.
"""

import re

# Health and diagnoses never become long-term memory. The prompt says so; this list is the second
# line of defense, in code. Whole words, with Hebrew prefix letters. Over-matching is fine: a note
# that is dropped by mistake is easy for the counselor to write by hand.
_SENSITIVE_WORDS = (
    "רופא רופאה רופאים תרופה תרופות תרופתי תרופתית כאב כאבי כאבים כואב כואבת כואבים מחלה מחלות חולה חולים "
    "חולת חלתה מחלת אבחנה אבחנות אבחון אובחן אובחנה אובחנו מאובחן מאובחנת הפרעה הפרעת הפרעות קשב דיכאון "
    "חרדה חרדות פסיכולוג פסיכולוגית פסיכולוגי פסיכיאטר פסיכיאטרית ריטלין אשפוז אושפז אושפזה "
    "מאושפז מאושפזת פציעה פצוע פצועה נפצע נפצעה תסמונת בריאות בריאותי בריאותית כדורים סוכרת "
    "אסטמה אפילפסיה ניתוח הריון בהריון סמים ADHD ADD OCD PTSD קב\"ן קב״ן קבן"
).split()
# Stems: every inflection of these ("פסיכולוגים", "מאובחנים", "רופאת", "נפשית", "אובדניות").
_SENSITIVE_STEMS = (
    "רופא תרופ פסיכולוג פסיכיאטר מאובחנ אובחנ אבחנ אבחונ אשפוז אושפז מאושפז נפצע פציע תסמונ בריאות "
    "נפשי אובדנ דיכאונ חרדת הריונ אפילפ רפוא מרפא"
).split()
_SENSITIVE_PHRASES = [
    r"(?:פגיע\w*|לפגוע|פוגע\w*|פגעה|פגע)\s+(?:ב)?עצמ\w*",  # self-harm
    r"טיפול\s+(?:רגשי|פסיכולוגי|תרופתי|נפשי)\w*",
    r"ד[\"״]ר",  # Dr.
    r"מצב\s+נפשי\w*|בריאות\s+הנפש",
]
_PREFIX = r"(?<!\w)[ובכלמשה]*"
_SENSITIVE = re.compile(
    "|".join(
        [
            rf"{_PREFIX}(?:{'|'.join(_SENSITIVE_WORDS)})(?!\w)",
            rf"{_PREFIX}(?:{'|'.join(_SENSITIVE_STEMS)})\w*",
            *_SENSITIVE_PHRASES,
        ]
    ),
    re.IGNORECASE,
)


def is_sensitive(text: str) -> bool:
    return _SENSITIVE.search(text) is not None


HEALTH_MASK = "[…]"


def mask_sensitive_words(text: str) -> tuple[str, int]:
    """The text with each health WORD replaced by HEALTH_MASK, and how many. For the safety screen
    only (ai/risk_screen.py): removing whole sentences there would hide a risk scene that happens to
    sit in a sentence about health ("the psychologist told me he stood on the roof at night"). The
    scene goes to the screen, the medical words do not. Everywhere else, whole sentences are removed."""
    return _SENSITIVE.subn(HEALTH_MASK, text)


# A sentence ends at . ! ? or a line break. Hebrew notes rarely use anything else.
_SENTENCE_END = re.compile(r"(?<=[.!?\n])")
# A sentence that only continues the one before it: removed together with a removed one
# ("יש לו ADHD. לכן הוא לוקח משהו כל בוקר."), and so is a list under a removed heading line.
_CONTINUES = re.compile(r"^\s*(?:ו?לכן|בגלל זה|בעקבות זה|בעקבות כך|כתוצאה|מסיבה זו|זה גם|וזה)(?!\w)")
_LIST_ITEM = re.compile(r"^\s*(?:[-•*]|\d+[.)])\s")


def remove_sensitive_sentences(notes: str) -> tuple[str, int]:
    """The notes without the sentences about health or a diagnosis, and how many were removed.
    Runs before anonymization and before anything is sent: what the model never receives, it can
    never repeat, summarize or propose to remember. Over-matching is the safe direction here too:
    a removed sentence costs a less complete summary, never a privacy leak."""
    kept: list[str] = []
    removed = 0
    previous_removed = False
    under_removed_heading = False
    for sentence in _SENTENCE_END.split(notes):
        if under_removed_heading and not _LIST_ITEM.match(sentence):
            under_removed_heading = False  # the list ended
        drop = (
            is_sensitive(sentence)
            or (previous_removed and _CONTINUES.match(sentence) is not None)
            or under_removed_heading
        )
        if drop:
            removed += 1
            kept.append("\n" if sentence.endswith("\n") else "")  # keep the line structure
            if sentence.rstrip().endswith(":"):
                under_removed_heading = True
        else:
            kept.append(sentence)
        if sentence.strip():
            previous_removed = drop
    return "".join(kept), removed
