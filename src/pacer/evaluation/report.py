"""A run, as a Markdown report to read and a JSON summary to compare runs by (task 9).

The report shows every answer in full, with the judge's reasons: the judge is a model too, and the
plan is that a person checks a sample of its verdicts by hand.
"""

from collections.abc import Callable, Sequence
from typing import Any

from .run import LibraryCaseResult, TalkCaseResult

COVERAGE = {"direct": "ישיר", "partial": "חלקי", "none": "אין"}


def _mark(value: bool | None) -> str:
    return "-" if value is None else ("✓" if value else "✗")


def _fraction(part: int, whole: int) -> str:
    return "-" if whole == 0 else f"{part}/{whole}"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.0%}"


def library_section(results: Sequence[LibraryCaseResult], summary: dict) -> list[str]:
    s = summary
    lines = [
        "## הספרייה",
        "",
        "| מדד | עם הספרייה | בלי (baseline) |",
        "|---|---|---|",
        f"| הכיסוי שדווח נכון | {_pct(s['coverage_ok'])} (בשאלות שנבדקו: {_pct(s['coverage_ok_reviewed'])}) | - |",
        f"| מקור נכון בין הקטעים שנשלפו | {_pct(s['retrieved_right'])} | - |",
        f"| מקור נכון צוטט | {_pct(s['cited_right'])} | - |",
        f"| טענות שהקטע שלהן באמת אומר אותן | {_pct(s['claims_supported'])} | - |",
        f"| סעיפי בדיקה שעברו | {_pct(s['checklist'])} | {_pct(s['checklist_baseline'])} |",
        f"| מועילות (1-5) | {s['helpful'] or '-'} | {s['helpful_baseline'] or '-'} |",
        "",
        f'- "אין מקור" כשיש מקור: {s["said_none_with_source"]}. ענה כשאין מקור: {s["answered_without_source"]}.',
        f"- ממצאי ולידציה בתשובות: {s['validation_issues']}. זמן ממוצע לתשובה: {s['seconds_per_answer']} שניות.",
        "",
        "| מקרה | סוג | כיסוי (צפוי) | שליפה | ציטוט | טענות | בדיקה | מועילות | baseline: בדיקה, מועילות |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        g, b = r.grade, r.baseline_grade
        expected = "/".join(COVERAGE[c] for c in r.case.coverage)
        coverage = f"{COVERAGE[r.answer.coverage]} ({expected}) {_mark(r.coverage_ok)}"
        claims = _fraction(g.claims_supported, len(g.claims)) if g else "-"
        checklist = _fraction(g.checklist_passed, len(g.checklist)) if g else "-"
        base = f"{_fraction(b.checklist_passed, len(b.checklist))}, {b.helpful}" if b else "-"
        review = "" if r.case.reviewed else " *"
        lines.append(
            f"| {r.case.id}{review} | {r.case.kind} | {coverage} | {_mark(r.retrieved_right)} | {_mark(r.cited_right)} "
            f"| {claims} | {checklist} | {g.helpful if g else '-'} | {base} |"
        )
    lines += ["", "\\* עוד לא נבדק על ידי Itay.", "", "### התשובות", ""]
    for r in results:
        lines += [f"<details><summary>{r.case.id}: {r.case.question}</summary>", "", r.answer.answer, ""]
        if r.answer.claims:
            lines += [f"- {c.text} [{', '.join(c.passages)}]" for c in r.answer.claims] + [""]
        if r.grade:
            lines += [f"**השופט:** {r.grade.why}", ""]
        if r.baseline:
            lines += ["**בלי הספרייה:**", "", r.baseline, ""]
            if r.baseline_grade:
                lines += [f"**השופט:** {r.baseline_grade.why}", ""]
        lines += ["</details>", ""]
    return lines


def talk_section(results: Sequence[TalkCaseResult], summary: dict) -> list[str]:
    variants = [v for v in ("masked", "words") if v in summary]
    names = {"masked": "עם ההתממה כמו שהיא", "words": "מילים נשארות מילים"}
    lines = ["## שיחות אישיות", "", "| מדד | " + " | ".join(names[v] for v in variants) + " |"]
    lines.append("|---|" + "---|" * len(variants))
    rows: list[tuple[str, str, Callable[[Any], str]]] = [
        ("מטרות חובה בטווח", "required_in_range", _pct),
        ("הצעות התקדמות מחוץ למפתח", "wrong_progress", str),
        ("הערות צפויות שנמצאו", "notes_found", _pct),
        ("...ומאותו סוג", "notes_kind_right", _pct),
        ("פרט רפואי בהצעה", "forbidden_found", str),
        ("ממצאי ולידציה", "validation_issues", str),
        ("שניות לשיחה", "seconds_per_talk", str),
    ]
    for label, key, fmt in rows:
        lines.append(f"| {label} | " + " | ".join(fmt(summary[v][key]) for v in variants) + " |")
    lines += ["", "| שיחה | גרסה | מטרות בטווח | מחוץ למפתח | הערות | רפואי |", "|---|---|---|---|---|---|"]
    for r in results:
        notes = _fraction(r.notes_found, len(r.talk.case.expected_notes))
        wrong = ", ".join(r.wrong_progress) or "-"
        lines.append(
            f"| {r.talk.case.id} | {names[r.variant]} | {_fraction(r.required_in_range, len(r.required))} "
            f"| {wrong} | {notes} | {', '.join(r.forbidden_found) or '-'} |"
        )
    lines += ["", "### ההצעות", ""]
    for r in results:
        lines += [f"<details><summary>{r.talk.case.id} ({names[r.variant]}): {r.talk.case.tests}</summary>", ""]
        lines += [r.summary, ""]
        lines += [f"- התקדמות: {goal} → {value}%" for goal, value in r.progress.items()]
        lines += [f"- הערה ({kind}): {text}" for kind, text in r.notes]
        lines += ["", "</details>", ""]
    return lines


def render(
    title: str,
    meta: Sequence[str],
    library: tuple[Sequence[LibraryCaseResult], dict] | None,
    talks: tuple[Sequence[TalkCaseResult], dict] | None,
) -> str:
    lines = [f"# {title}", "", *[f"- {m}" for m in meta], ""]
    if library:
        lines += library_section(*library)
    if talks:
        lines += talk_section(*talks)
    return "\n".join(lines)
