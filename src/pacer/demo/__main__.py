"""Command line: load the demo academy into the database configured in .env.

uv run python -m pacer.demo            # refuses if the database already has data
uv run python -m pacer.demo --reset    # deletes the academy's data first (development only); the library stays
"""

import argparse
import logging

from sqlalchemy import Engine, text

from pacer.logging_config import configure_logging
from pacer.repositories import get_engine, repositories
from pacer.repositories.tables import Base, StaffMemberRow

from .academy import Summary, build_academy

log = logging.getLogger("pacer.demo")  # not "__main__" when run with -m


def database_is_empty(engine: Engine) -> bool:
    with engine.connect() as connection:
        return bool(
            connection.execute(text(f"SELECT NOT EXISTS (SELECT 1 FROM {StaffMemberRow.__tablename__})")).scalar()
        )


# The library is not demo data: it is imported from the academy's own files and indexed with a local
# model (python -m pacer.library import), which takes minutes. A reset of the demo academy keeps it.
# (Found 2026-10-27: --reset emptied it, and every library question got "no source".) No table here
# references the academy's tables or is referenced by them, so CASCADE never reaches them.
LIBRARY_TABLES = frozenset({"knowledge_sources", "chunks", "chunk_embeddings"})


def reset_database(engine: Engine) -> None:
    """Empty every table of the academy and restart the ids at 1. The tables stay, and so does the library."""
    tables = ", ".join(t.name for t in Base.metadata.sorted_tables if t.name not in LIBRARY_TABLES)
    with engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


def load(engine: Engine, *, reset: bool, seed: int) -> Summary:
    if reset:
        log.warning("Deleting all data (--reset)")
        reset_database(engine)
    elif not database_is_empty(engine):
        raise SystemExit(
            "The database already has data. Run with --reset to replace it (deletes everything)."
        )

    with repositories(engine) as repos:  # all or nothing: a failure leaves the database untouched
        return build_academy(repos, seed=seed)


def print_summary(summary: Summary) -> None:
    print(
        f"\nLoaded: {len(summary.trainees)} trainees, {summary.goal_count} goals, "
        f"{summary.update_count} progress updates, {summary.note_count} approved notes, "
        f"{len(summary.approvals)} approval requests, {summary.slot_count} weekly slots, "
        f"{summary.session_count} sessions, {summary.attendance_count} attendance marks, {summary.task_count} tasks"
    )
    print("Trainee statuses:", ", ".join(f"{s}={n}" for s, n in sorted(summary.trainee_statuses.items())))
    print("\nTry the API at http://127.0.0.1:8000/docs with X-Staff-Id:")
    print(f"  {summary.manager.id}  {summary.manager.full_name} (manager)")
    for group in summary.groups:
        for counselor in summary.counselors:
            if counselor.id in group.counselor_ids:
                print(f"  {counselor.id}  {counselor.full_name} (counselor, {group.name})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m pacer.demo", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--reset", action="store_true", help="delete the academy's data before loading (the library stays)"
    )
    parser.add_argument("--seed", type=int, default=2026, help="a different seed makes a different academy")
    args = parser.parse_args(argv)

    configure_logging()
    print_summary(load(get_engine(), reset=args.reset, seed=args.seed))


if __name__ == "__main__":
    main()
