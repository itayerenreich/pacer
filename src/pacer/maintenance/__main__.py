"""Command line: maintenance on the database configured in .env.

uv run python -m pacer.maintenance purge-notes              # shows what would be deleted
uv run python -m pacer.maintenance purge-notes --yes        # deletes it
uv run python -m pacer.maintenance purge-notes --days 180   # a shorter period than the default 365

purge-notes deletes the notes (long-term memory) and the personal talks of trainees whose file was
closed (left or graduated) longer ago than the period. Without --yes nothing is deleted.
"""

import argparse
import sys
from datetime import UTC, datetime, timedelta

from pacer.logging_config import configure_logging
from pacer.repositories import get_engine, repositories
from pacer.services.retention import DEFAULT_KEEP_FOR, MAX_KEEP_FOR, past_retention, purge


def main(argv: list[str] | None = None, *, now: datetime | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pacer.maintenance", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    purge_parser = commands.add_parser(
        "purge-notes", help="delete notes and talks of files closed longer ago than --days"
    )
    purge_parser.add_argument("--days", type=int, default=DEFAULT_KEEP_FOR.days, help="the retention period")
    purge_parser.add_argument("--yes", action="store_true", help="really delete (otherwise only show)")
    args = parser.parse_args(argv)
    if not 0 <= args.days <= MAX_KEEP_FOR.days:
        parser.error(f"--days must be between 0 and {MAX_KEEP_FOR.days}")

    configure_logging()
    now, keep_for = now or datetime.now(UTC), timedelta(days=args.days)
    with repositories(get_engine()) as repos:  # one transaction: all or nothing
        plan = past_retention(repos.trainees, repos.notes, repos.talks, repos.tasks, now=now, keep_for=keep_for)
        done = (
            purge(repos.trainees, repos.notes, repos.talks, repos.tasks, plan, now=now, keep_for=keep_for)
            if args.yes
            else None
        )
    # Printed only after the block: by now the transaction has committed (or raised).
    print(
        f"{plan.note_count} notes, {plan.talk_count} talks and {plan.task_count} tasks "
        f"of {len(plan.trainee_ids)} trainees are past {args.days} days"
    )
    if done is not None and done.trainee_ids:
        print(f"Deleted {done.note_count} notes, {done.talk_count} talks and {done.task_count} tasks.")
    elif plan.trainee_ids:
        print("Nothing was deleted. Run again with --yes to delete them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
