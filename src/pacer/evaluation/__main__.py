"""Command line: the evaluation sets.

uv run python -m pacer.evaluation check                               # are the sets valid, and do they fit?
uv run python -m pacer.evaluation masking --out data/masking-report.md  # anonymization vs the answer key

Both read the academy from the database in .env (the demo academy: python -m pacer.demo), and need
no model. Library retrieval is measured by: python -m pacer.library compare evals/library-v1.json
"""

import argparse
from pathlib import Path

from pacer.ai.anonymization import load_roster
from pacer.library.__main__ import as_document
from pacer.library.manifest import load_manifest
from pacer.logging_config import configure_logging, use_utf8_console
from pacer.repositories import get_engine, repositories

from .library_set import load_library_set, unknown_sources
from .masking import MentionResult, Outcome, measure, summary
from .talk_set import CaseError, load_talk_set, resolve

LIBRARY_SET = Path("evals/library-v1.json")
TALK_SET = Path("evals/talks-v1.json")
MANIFESTS = [Path("library/classics.json")]


def _uow():
    engine = get_engine()
    return lambda: repositories(engine)


LABELS = {
    Outcome.PERSON_RIGHT: "אדם, הטוקן הנכון",
    Outcome.PERSON_NEUTRAL: "אדם, טוקן ניטרלי ([NAME_n])",
    Outcome.PERSON_WRONG: "אדם, טוקן של אדם אחר",
    Outcome.PERSON_LEAKED: "אדם, השם נשלח כמו שהוא (דליפה!)",
    Outcome.WORD_KEPT: "מילה, נשארה מילה",
    Outcome.WORD_NEUTRAL: "מילה, הוסתרה כשם ([NAME_n])",
    Outcome.WORD_AS_PERSON: "מילה, נקשרה לחניך אמיתי",
}


def check() -> str:
    lines = []
    library = load_library_set(LIBRARY_SET)
    titles = {s.title for m in MANIFESTS for s in load_manifest(m).sources}
    missing = unknown_sources(library, titles)
    reviewed = sum(c.reviewed for c in library.cases)
    lines.append(f"{library.ref}: {len(library.cases)} cases ({reviewed} reviewed)")
    lines += [f"  ✗ no such source in the library: {t}" for t in missing] or ["  ✓ every expected source exists"]
    talks = load_talk_set(TALK_SET)
    lines.append(f"{talks.ref}: {len(talks.cases)} cases ({sum(c.reviewed for c in talks.cases)} reviewed)")
    errors = []
    with _uow()() as repos:
        for case in talks.cases:
            try:
                resolve(case, repos.staff, repos.trainees, repos.goals)
            except CaseError as error:
                errors.append(str(error))
    lines += [f"  ✗ {e}" for e in errors] or ["  ✓ every case fits the academy in the database"]
    return "\n".join(lines)


def _line(r: MentionResult) -> str:
    return f"- `{r.mention.word}` ({r.mention.occurrence}) · {r.mention.means} → `{r.sent_as}` · {LABELS[r.outcome]}"


def masking() -> str:
    talks = load_talk_set(TALK_SET)
    results = []
    details = ["## לפי שיחה", ""]
    with _uow()() as repos:
        roster = load_roster(repos.staff, repos.trainees)
        for case in talks.cases:
            measured = measure(resolve(case, repos.staff, repos.trainees, repos.goals), roster)
            results += measured
            details += [f"**{case.id}**: {case.tests}", ""]
            details += [_line(r) for r in measured] + [""]
    counts = summary(results)
    people = sum(n for o, n in counts.items() if o.startswith("person"))
    words = sum(n for o, n in counts.items() if o.startswith("word"))
    table = ["| מה קרה | כמה |", "|---|---|"] + [f"| {LABELS[o]} | {counts.get(o, 0)} |" for o in Outcome]
    head = [
        f"# התממה מול מפתח התשובות ({talks.ref})",
        "",
        f"{people} אזכורים של אנשים, {words} מילים שהן גם שמות. בלי מודל: זו בדיוק ההודעה שהייתה נשלחת.",
        "",
        *table,
        "",
    ]
    return "\n".join(head + details)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pacer.evaluation", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    masking_parser = commands.add_parser("masking")
    masking_parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    use_utf8_console()
    configure_logging()
    report = check() if args.command == "check" else masking()
    print(report)
    if getattr(args, "out", None):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(as_document(report), encoding="utf-8")
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
