"""Try the meeting pipeline on a sample transcript, against the demo academy in your database.

    uv run python -m pacer.demo.meeting weekly                 # dry run: shows EXACTLY what would be sent
    uv run python -m pacer.demo.meeting weekly --send          # sends it to Claude and saves the proposals
    uv run python -m pacer.demo.meeting short --send --out data/short.md   # also write the result to a file

Needs the demo academy (uv run python -m pacer.demo --reset). The dry run changes nothing, costs
nothing and needs no API key. --send records the meeting as the manager, runs the pipeline (about
a tenth of a cent with Haiku), and prints the summary, the proposals and every validation issue.
Then approve or reject the proposals at http://127.0.0.1:8000/docs (X-Staff-Id: 1).

Hebrew in a terminal may show in the wrong direction (a display issue only). --out writes a UTF-8
Markdown file: open it in VS Code (Ctrl+Shift+V) to read it properly. data/ is outside Git.
"""

import argparse
from pathlib import Path

from pacer.ai import LLMError
from pacer.ai.anonymization import load_roster
from pacer.ai.meeting_pipeline import MeetingPipeline, anonymize_meeting, extraction_message
from pacer.ai.routing import AITask, get_router
from pacer.logging_config import configure_logging, use_utf8_console
from pacer.repositories import get_engine, repositories
from pacer.services import MeetingService

from .meetings import SAMPLES, SampleMeeting

MANAGER_ID = 1


def dry_run(sample: SampleMeeting) -> str:
    with repositories(get_engine()) as repos:
        roster = load_roster(repos.staff, repos.trainees)
        participants = [s for i in sample.participant_ids if (s := repos.staff.get(i)) is not None]
    if not roster.people:
        raise SystemExit("The database is empty. Load the demo first: uv run python -m pacer.demo --reset")
    safe, removed = anonymize_meeting(roster, sample.transcript, participants)  # the pipeline's own code path
    lines = [
        f"# Dry run: {sample.title}",
        "",
        "Nothing was sent. This is the exact message the cloud model would receive:",
        "",
        "```",
        extraction_message(safe, sample.held_on, participants),
        "```",
        "",
        f"People replaced by tokens: {len(safe.people)}. Neutral tokens (ambiguous names, or words): "
        f"{len(safe.unresolved)}. Sentences about health removed: {removed}.",
    ]
    return "\n".join(lines)


def send(sample: SampleMeeting) -> str:
    try:  # before recording anything: is Claude configured? (a missing key fails here, not halfway)
        get_router().client_for(AITask.EXTRACT_MEETING)
    except LLMError as error:
        raise SystemExit(f"Cannot send: {error}") from error
    engine = get_engine()
    with repositories(engine) as repos:
        manager = repos.staff.get(MANAGER_ID)
        if manager is None:
            raise SystemExit("The database is empty. Load the demo first: uv run python -m pacer.demo --reset")
        meeting = MeetingService(repos.meetings, repos.proposals, repos.tasks, repos.staff).create_meeting(
            manager,
            title=sample.title,
            held_on=sample.held_on,
            participant_ids=sample.participant_ids,
            transcript=sample.transcript,
        )
        names = {s.id: s.full_name for s in repos.staff.list_all()}

    result = MeetingPipeline(lambda: repositories(engine), get_router()).process(manager, meeting.key)

    lines = [f"# {sample.title} (meeting {meeting.id})", "", "## Summary", "", result.meeting.summary or "", ""]
    lines += ["## Proposed tasks (waiting for the manager)", ""]
    for p in result.proposals:
        owner = names.get(p.owner_id, "(no owner: choose one before approving)")
        due = p.due_date.isoformat() if p.due_date else "no date"
        lines.append(f"- [{p.id}] {p.description}  |  {owner}  |  {due}")
    lines += ["", "## Validation issues", ""]
    lines += [f"- item {i.item or '-'}: {i.code}: {i.detail}" for i in result.issues] or ["- none"]
    cost = "unknown" if result.cost_usd is None else f"${result.cost_usd:.5f}"
    lines += ["", f"Prompt {result.prompt_ref}, attempts {result.attempts}, cost {cost}."]
    lines += [f"Approve or reject at http://127.0.0.1:8000/docs with X-Staff-Id: {MANAGER_ID}."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pacer.demo.meeting", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sample", choices=sorted(SAMPLES), help="which sample transcript")
    parser.add_argument("--send", action="store_true", help="really send it to Claude (costs a fraction of a cent)")
    parser.add_argument("--out", type=Path, help="also write the result to this UTF-8 Markdown file")
    args = parser.parse_args(argv)

    use_utf8_console()
    configure_logging()
    sample = SAMPLES[args.sample]
    report = send(sample) if args.send else dry_run(sample)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
