"""Try the personal talk pipeline on the sample talk, against the demo academy in your database.

    uv run python -m pacer.demo.talk maya                    # dry run: shows EXACTLY what would be sent
    uv run python -m pacer.demo.talk maya --send             # sends it to Claude and saves the proposals
    uv run python -m pacer.demo.talk maya --send --out data/maya.md

Needs the demo academy (uv run python -m pacer.demo --reset). The dry run changes nothing and
needs no API key. --send records the talk as the trainee's counselor and runs the summary. Then
approve or reject at http://127.0.0.1:8000/docs (X-Staff-Id: the counselor's id, printed below).
"""

import argparse
import sys
from pathlib import Path

from pacer.ai import LLMError
from pacer.ai.anonymization import load_roster
from pacer.ai.routing import AITask, get_router
from pacer.ai.talk_pipeline import TalkPipeline, anonymize_talk, talk_service
from pacer.logging_config import configure_logging, use_utf8_console
from pacer.repositories import get_engine, repositories

from .talks import SAMPLE_TALKS, SampleTalk


def dry_run(sample: SampleTalk) -> str:
    with repositories(get_engine()) as repos:
        trainee, counselor = repos.trainees.get(sample.trainee_id), repos.staff.get(sample.counselor_id)
        if trainee is None or counselor is None:
            raise SystemExit("Load the demo first: uv run python -m pacer.demo --reset")
        goals = repos.goals.list_by_trainee(trainee.id)
        roster = load_roster(repos.staff, repos.trainees)
    safe = anonymize_talk(roster, trainee, counselor, sample.held_on, goals, sample.notes)  # the pipeline's own path
    return "\n".join(
        [
            "# Dry run: personal talk",
            "",
            "Nothing was sent. This is the exact message the cloud model would receive:",
            "",
            "```",
            safe.text,
            "```",
        ]
    )


def send(sample: SampleTalk) -> str:
    try:
        get_router().client_for(AITask.SUMMARIZE_TALK)
    except LLMError as error:
        raise SystemExit(f"Cannot send: {error}") from error
    engine = get_engine()
    with repositories(engine) as repos:
        counselor = repos.staff.get(sample.counselor_id)
        if counselor is None:
            raise SystemExit("Load the demo first: uv run python -m pacer.demo --reset")
        talk = talk_service(repos).record_talk(counselor, sample.trainee_id, held_on=sample.held_on, notes=sample.notes)
        goals = {g.id: g.title for g in repos.goals.list_by_trainee(sample.trainee_id)}

    result = TalkPipeline(lambda: repositories(engine), get_router()).summarize(counselor, talk.id)
    lines = [f"# Talk {talk.id}", "", "## Summary", "", result.talk.summary, "", "## Proposals", ""]
    for p in result.proposals:
        if p.goal_id is not None:
            lines.append(f"- [{p.id}] progress: {goals[p.goal_id]} -> {p.progress}%  |  {p.text}")
        else:
            lines.append(f"- [{p.id}] note ({p.note_kind}): {p.text}")
    lines += ["", "## Validation issues", ""]
    lines += [f"- {i.section or 'summary'} {i.item or ''}: {i.code}: {i.detail}" for i in result.issues] or ["- none"]
    cost = "unknown" if result.cost_usd is None else f"${result.cost_usd:.5f}"
    lines += ["", f"Prompt {result.prompt_ref}, attempts {result.attempts}, cost {cost}."]
    lines += [f"Decide at http://127.0.0.1:8000/docs with X-Staff-Id: {counselor.id}."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pacer.demo.talk", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("sample", choices=sorted(SAMPLE_TALKS))
    parser.add_argument("--send", action="store_true", help="really send it to Claude (a fraction of a cent)")
    parser.add_argument("--out", type=Path, help="also write the result to this UTF-8 Markdown file")
    args = parser.parse_args(argv)

    use_utf8_console()
    configure_logging()
    sample = SAMPLE_TALKS[args.sample]
    report = send(sample) if args.send else dry_run(sample)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
