"""Try the counselor coach on the demo academy in your database.

    uv run python -m pacer.demo.coach "נדב הררי" "איך לעזור לו להשתתף יותר בשיחות הקבוצה?"          # dry run
    uv run python -m pacer.demo.coach "נדב הררי" "איך לעזור לו להשתתף יותר בשיחות הקבוצה?" --send
    uv run python -m pacer.demo.coach - "איך מתמודדים עם קבוצה שמאחרת?" --send --out data/coach.md  # general

The trainee is a full name from the demo, or "-" for a general question. The question is asked as the
trainee's primary counselor (or --as a staff id). The dry run sends nothing and needs no API key: it
shows the phrase-list check and EXACTLY the text the coach would send (passages are added at --send,
when Ollama searches the library). --send asks the real models (the safety screen and the coach) and
needs ANTHROPIC_API_KEY and Ollama with bge-m3. Nothing is written to the database.
"""

import argparse
from pathlib import Path

from pacer.ai import LLMError
from pacer.ai.anonymization import load_roster
from pacer.ai.coach import Coach, CoachOutcome, CoachResult, prepare_coach_message
from pacer.ai.embeddings import DEFAULT_EMBEDDING_MODEL, ollama_embedder
from pacer.ai.routing import get_router
from pacer.ai.safety import check_safety
from pacer.domain import StaffMember
from pacer.evaluation.coach_eval import advice_text
from pacer.library.__main__ import as_document
from pacer.library.chunking import DEFAULT_CHUNKING
from pacer.logging_config import configure_logging, use_utf8_console
from pacer.repositories import get_engine, repositories
from pacer.services import CoachContextService


def _find(trainee_name: str, staff_id: int | None) -> tuple[StaffMember, int | None]:
    with repositories(get_engine()) as repos:
        trainee = None
        if trainee_name != "-":
            trainee = next((t for t in repos.trainees.list_all() if t.full_name == trainee_name), None)
            if trainee is None:
                raise SystemExit(f"No trainee named {trainee_name}. Load the demo: uv run python -m pacer.demo --reset")
        asker_id = staff_id or (trainee.primary_counselor_id if trainee else 1)
        asker = repos.staff.get(asker_id)
        if asker is None:
            raise SystemExit(f"No staff member with id {asker_id}")
        return asker, trainee.key if trainee else None


def dry_run(question: str, asker: StaffMember, trainee_id: int | None) -> str:
    with repositories(get_engine()) as repos:
        context = None
        if trainee_id is not None:
            context = CoachContextService(
                repos.trainees, repos.groups, repos.goals, repos.goal_updates, repos.notes
            ).build(asker, trainee_id)
        roster = load_roster(repos.staff, repos.trainees)
    found = check_safety(question)
    if found.is_risk:
        return "# Dry run\n\nThe phrase list finds an exceptional incident: the procedure, and nothing is sent."
    prepared = prepare_coach_message(roster, question, context)  # the coach's own code path
    return "\n".join(
        [
            "# Dry run: the counselor coach",
            "",
            f"Asked by: {asker.full_name} (staff {asker.id}). Phrase list: no exceptional incident.",
            f"Health sentences removed before sending: {prepared.removed}.",
            "With --send, the safety screen reads the question first; then the coach receives, anonymized:",
            "",
            "```",
            prepared.document.text,
            "```",
            "",
            "(and the library passages closest to the question)",
        ]
    )


def render(result: CoachResult) -> str:
    title = {
        CoachOutcome.ADVICE: "עצה",
        CoachOutcome.REFERRAL: "אירוע חריג: נוהל",
        CoachOutcome.NOT_SENT: "לא נשלח",
    }[result.outcome]
    lines = [f"# המלווה: {title}", "", f"**שאלה:** {result.question}", "", advice_text(result)]
    if result.referral is not None:
        lines += ["", *[f"- {s.contact}" for s in result.referral.steps if s.contact]]
    if result.advice is not None:
        cited = [c for c in result.advice.citations if c.cited]
        lines += ["", f"קטעים מהספרייה שנשלחו למלווה: {len(result.advice.citations)}, מהם צוטטו: {len(cited)}"]
        if cited:
            lines += ["", "**מקורות:**", *[f"- {c.passage}: {c.source} · {c.evidence}" for c in cited]]
        if result.advice.issues:
            lines += ["", "**מה הוולידציה תיקנה:** " + "; ".join(f"{i.code}: {i.detail}" for i in result.advice.issues)]
    lines += ["", f"_{result.note}_", "", f"prompt: {result.prompt_ref or '-'} · cost: {result.cost_usd}"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pacer.demo.coach", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("trainee", help='a demo trainee\'s full name, or "-" for a general question')
    parser.add_argument("question")
    parser.add_argument("--as", dest="staff_id", type=int, help="ask as this staff id (default: the primary counselor)")
    parser.add_argument("--send", action="store_true", help="ask the real models")
    parser.add_argument("--out", type=Path, help="also write the answer to this Markdown file")
    args = parser.parse_args(argv)

    use_utf8_console()
    configure_logging()
    asker, trainee_id = _find(args.trainee, args.staff_id)
    if not args.send:
        report = dry_run(args.question, asker, trainee_id)
    else:
        coach = Coach(
            lambda: repositories(get_engine()), get_router(), ollama_embedder(DEFAULT_EMBEDDING_MODEL), DEFAULT_CHUNKING
        )
        try:
            report = render(coach.ask(asker, args.question, trainee_id))
        except LLMError as error:
            reason = f"A model could not be used ({type(error).__name__}): nothing was answered. {error}"
            raise SystemExit(reason) from None
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(as_document(report), encoding="utf-8")  # right-to-left in VS Code's preview
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
