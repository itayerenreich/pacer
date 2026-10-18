"""Note retention against PostgreSQL: closed_at is stored, the database enforces the rule too,
and the command line shows by default and deletes only with --yes."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

import pacer.maintenance.__main__ as maintenance
from pacer.demo.__main__ import load
from pacer.domain import NoteKind, TraineeNote, TraineeStatus
from pacer.repositories import repositories

pytestmark = pytest.mark.integration


def note_count(engine) -> int:
    with engine.connect() as connection:
        return connection.execute(text("SELECT count(*) FROM trainee_notes")).scalar()


@pytest.fixture
def academy(engine, monkeypatch):
    summary = load(engine, reset=False, seed=2026)
    monkeypatch.setattr(maintenance, "get_engine", lambda: engine)
    return summary


def test_the_demo_trainee_who_left_has_a_closed_at(engine, academy):
    with repositories(engine) as repos:
        left = [t for t in repos.trainees.list_all() if t.status is TraineeStatus.LEFT]
    assert len(left) == 1 and left[0].closed_at is not None


def test_the_database_rejects_a_closed_status_without_a_date(engine, academy):
    with pytest.raises(IntegrityError, match="ck_trainees_closed_at_matches_status"), engine.begin() as connection:
        connection.execute(text("UPDATE trainees SET status = 'left', closed_at = NULL WHERE id = 1"))


def test_purge_notes_shows_first_and_deletes_only_with_yes(engine, academy, capsys):
    with repositories(engine) as repos:
        trainee = next(t for t in repos.trainees.list_all() if t.status is TraineeStatus.LEFT)
        repos.notes.add(TraineeNote(trainee_id=trainee.id, kind=NoteKind.OBSERVATION, text="מוביל שיחות בקבוצה", author_id=1))
    later = trainee.closed_at + timedelta(days=400)

    maintenance.main(["purge-notes"], now=later)
    assert "1 notes of 1 trainees" in capsys.readouterr().out
    assert note_count(engine) == 1  # only shown

    maintenance.main(["purge-notes", "--days", "500", "--yes"], now=later)
    assert note_count(engine) == 1  # within a longer period: kept

    maintenance.main(["purge-notes", "--yes"], now=later)
    assert note_count(engine) == 0


def test_purge_with_nothing_to_delete_says_so(engine, academy, capsys):
    maintenance.main(["purge-notes", "--yes"], now=datetime.now(UTC))
    assert "0 notes" in capsys.readouterr().out
