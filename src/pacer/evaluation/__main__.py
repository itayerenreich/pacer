"""Command line: the evaluation sets.

uv run python -m pacer.evaluation check                               # are the sets valid, and do they fit?
uv run python -m pacer.evaluation masking --out data/masking-report.md  # anonymization vs the answer key
uv run python -m pacer.evaluation run --out data/eval-claude-haiku.md     # REAL models: costs about $1
uv run python -m pacer.evaluation run --model claude-sonnet --only talks --limit 2 --out data/try.md

check and masking need no model. run needs ANTHROPIC_API_KEY, and Ollama for the embeddings (bge-m3).
All read the academy from the database in .env (the demo academy: python -m pacer.demo). run writes the
report (Markdown) and, next to it, a JSON summary that task 9 compares runs by.
Library retrieval alone (no answering model): python -m pacer.library compare evals/library-v1.json
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from pacer.ai import LLMError
from pacer.ai.anonymization import Anonymizer, load_roster
from pacer.ai.answer import PROMPT_VERSION as ANSWER_PROMPT_VERSION
from pacer.ai.answer import LibraryAnswerer
from pacer.ai.embeddings import DEFAULT_EMBEDDING_MODEL, ollama_embedder
from pacer.ai.routing import MODELS, AITask
from pacer.ai.structured import StructuredOutputError
from pacer.ai.talk_pipeline import PROMPT_VERSION as TALK_PROMPT_VERSION
from pacer.library.__main__ import as_document
from pacer.library.chunking import DEFAULT_CHUNKING
from pacer.library.manifest import load_manifest
from pacer.logging_config import configure_logging, use_utf8_console
from pacer.repositories import get_engine, repositories

from .library_set import load_library_set, unknown_sources
from .masking import MentionResult, Outcome, measure, summary
from .report import render
from .run import (
    LibraryCaseResult,
    TalkCaseResult,
    library_summary,
    router_for,
    run_library_case,
    run_talk_case,
    talk_summary,
)
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
                resolve(case, repos.staff, repos.trainees, repos.goals, repos.groups)
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
            measured = measure(resolve(case, repos.staff, repos.trainees, repos.goals, repos.groups), roster)
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


def _failure(case_id: str, error: Exception) -> str:
    """One line for a case that failed. The cause's message too: it says what the provider refused."""
    cause = f" ({type(error.__cause__).__name__}: {error.__cause__})" if error.__cause__ else ""
    if isinstance(error, StructuredOutputError):  # what was wrong with the answers, attempt by attempt
        cause += " [" + " | ".join(error.reasons)[:400] + "]"
    print(f"  ✗ {case_id} failed: {type(error).__name__}: {error}{cause}", flush=True)
    return f"{case_id}: {type(error).__name__}: {str(error)[:200]}{cause[:300]}"


def run(model: str, *, only: str | None, limit: int | None, baseline: bool, judge: bool) -> tuple[str, dict]:
    """Every case against real models. Returns the report and the summary (for the JSON file)."""
    router = router_for(model)
    uow = _uow()
    judge_model = router.route(AITask.JUDGE).model
    library_results: list[LibraryCaseResult] = []
    talk_results: list[TalkCaseResult] = []
    failed: list[str] = []  # a case whose model call failed is reported, and the run goes on
    library = load_library_set(LIBRARY_SET)
    talks = load_talk_set(TALK_SET)
    with uow() as repos:
        roster = load_roster(repos.staff, repos.trainees)
        titles = {s.key: s.title for s in repos.sources.list_all()}
        resolved = [resolve(c, repos.staff, repos.trainees, repos.goals, repos.groups) for c in talks.cases[:limit]]
    anonymizer = Anonymizer(roster)
    if only in (None, "library"):
        answerer = LibraryAnswerer(uow, router, ollama_embedder(DEFAULT_EMBEDDING_MODEL), DEFAULT_CHUNKING)
        for case in library.cases[:limit]:
            print(f"library {case.id} ...", flush=True)
            try:
                library_results.append(
                    run_library_case(case, answerer, router, anonymizer, titles, baseline=baseline, judge=judge)
                )
            except (LLMError, StructuredOutputError) as error:
                failed.append(_failure(case.id, error))
    if only in (None, "talks"):
        for talk in resolved:
            for words in (False, True):
                print(f"talk {talk.case.id} ({'words' if words else 'masked'}) ...", flush=True)
                try:
                    talk_results.append(run_talk_case(talk, roster, router, words_stay_words=words, judge=judge))
                except (LLMError, StructuredOutputError) as error:
                    failed.append(_failure(f"{talk.case.id} ({'words' if words else 'masked'})", error))
    lib = library_summary(library_results) if library_results else None
    tlk = talk_summary(talk_results) if talk_results else None
    sets = [library.ref, talks.ref]
    prompts = [f"library_answer@v{ANSWER_PROMPT_VERSION}", f"talk_summary@v{TALK_PROMPT_VERSION}"]
    summaries = {
        "model": model,
        "judge": judge_model if judge else None,
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "sets": sets,
        "prompts": prompts,
        "retrieval": [DEFAULT_EMBEDDING_MODEL, DEFAULT_CHUNKING],
        "library": lib,
        "talks": tlk,
        "failed": failed,
    }
    costs = ([f"library ${lib['cost_usd']}"] if lib else []) + [
        f"talks/{k} ${v['cost_usd']}" for k, v in (tlk or {}).items()
    ]
    self_judged = (
        " (אותו מודל: הוא שופט את עצמו, והציון עלול להיות מוטה כלפי מעלה)" if judge and judge_model == model else ""
    )
    meta = [
        f"מודל עונה: **{model}**. שופט: **{judge_model if judge else 'בלי'}**{self_judged}",
        f"סטים: {', '.join(sets)}. prompts: {', '.join(prompts)}",
        f"שליפה: {DEFAULT_EMBEDDING_MODEL} · {DEFAULT_CHUNKING}. עלות: {', '.join(costs)}",
    ]
    if failed:
        meta.append(f"**נכשלו ({len(failed)}), ולא נכללים בציונים:** " + "; ".join(failed))
    report = render(
        f"הערכה: {model}",
        meta,
        (library_results, lib) if lib else None,
        (talk_results, tlk) if tlk else None,
    )
    return report, summaries


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pacer.evaluation", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    masking_parser = commands.add_parser("masking")
    masking_parser.add_argument("--out", type=Path)
    running = commands.add_parser("run", help="every case against real models (costs money)")
    running.add_argument("--model", default="claude-haiku", choices=list(MODELS), help="the model that answers")
    running.add_argument("--only", choices=["library", "talks"])
    running.add_argument("--limit", type=int, help="only the first N cases of each set (a quick try)")
    running.add_argument("--no-baseline", action="store_true", help="skip answering without the library")
    running.add_argument("--no-judge", action="store_true", help="skip the judge (code checks only)")
    running.add_argument("--out", type=Path, required=True, help="the report; the JSON summary goes next to it")
    args = parser.parse_args(argv)

    use_utf8_console()
    configure_logging()
    summaries = None
    if args.command == "check":
        report = check()
    elif args.command == "masking":
        report = masking()
    else:
        report, summaries = run(
            args.model, only=args.only, limit=args.limit, baseline=not args.no_baseline, judge=not args.no_judge
        )
    print(report)
    if getattr(args, "out", None):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(as_document(report), encoding="utf-8")
        print(f"\nWritten to {args.out}")
        if summaries is not None:
            summary_file = args.out.with_suffix(".json")
            summary_file.write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Summary in {summary_file}")


if __name__ == "__main__":
    main()
