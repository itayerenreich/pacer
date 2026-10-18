"""Command line: maintenance on the database configured in .env.

uv run python -m pacer.maintenance purge-notes              # shows what would be deleted
uv run python -m pacer.maintenance purge-notes --yes        # deletes it
uv run python -m pacer.maintenance purge-notes --days 180   # a shorter period than the default 365

purge-notes deletes the notes (long-term memory) of trainees whose file was closed (left or
graduated) longer ago than the period. Without --yes nothing is deleted.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta

from pacer.logging_config import configure_logging
from pacer.repositories import get_engine, repositories
from pacer.services.retention import DEFAULT_KEEP_NOTES_FOR, notes_past_retention, purge_notes

log = logging.getLogger("pacer.maintenance")


def main(argv: list[str] | None = None, *, now: datetime | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pacer.maintenance", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    purge = commands.add_parser("purge-notes", help="delete notes of trainees closed longer ago than --days")
    purge.add_argument("--days", type=int, default=DEFAULT_KEEP_NOTES_FOR.days, help="the retention period")
    purge.add_argument("--yes", action="store_true", help="really delete (otherwise only show)")
    args = parser.parse_args(argv)
    if args.days < 0:
        parser.error("--days cannot be negative")

    configure_logging()
    with repositories(get_engine()) as repos:  # one transaction: all or nothing
        plan = notes_past_retention(
            repos.trainees, repos.notes, now=now or datetime.now(UTC), keep_for=timedelta(days=args.days)
        )
        print(f"{plan.note_count} notes of {len(plan.notes_by_trainee)} trainees are past {args.days} days")
        if args.yes and plan.note_count:
            purge_notes(repos.notes, plan)
            print("Deleted.")
        elif plan.note_count:
            print("Nothing was deleted. Run again with --yes to delete them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
