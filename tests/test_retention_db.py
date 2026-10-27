"""Note retention against PostgreSQL: closed_at is stored, the database enforces the rule too,
and the command line shows by default and deletes only with --yes."""

from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

import pacer.maintenance.__main__ as maintenance
from pacer.demo.__main__ import load
from pacer.domain import NoteKind, PersonalTalk, TalkProposal, TalkProposalKind, Task, TraineeNote, TraineeStatus
from pacer.repositories import repositories

pytestmark = pytest.mark.integration


def count(engine, table: str) -> int:
    with engine.connect() as connection:
        return connection.execute(text(f"SELECT count(*) FROM {table}")).scalar()


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
        repos.notes.add(TraineeNote(trainee_id=trainee.key, kind=NoteKind.OBSERVATION, text="מוביל שיחות בקבוצה", author_id=1))
        talk = repos.talks.add(PersonalTalk(trainee_id=trainee.key, counselor_id=2, held_on=date(2026, 9, 1), notes="שיחה על הקבוצה ועל מה שעבד השבוע."))
        repos.talk_proposals.add(TalkProposal(talk_id=talk.key, kind=TalkProposalKind.NOTE, note_kind=NoteKind.OBSERVATION, text="לזכור את זה"))
        repos.tasks.add(Task(description="לדבר איתו על החזרה הביתה", owner_id=2, created_by_id=2, trainee_id=trainee.key))
        tasks_before = len(repos.tasks.list_all())
    later = trainee.closed_at + timedelta(days=400)

    maintenance.main(["purge-notes"], now=later)
    assert "1 notes, 1 talks and 1 tasks of 1 trainees" in capsys.readouterr().out
    demo = academy.note_count  # the demo's own notes belong to active trainees: never purged
    assert count(engine, "trainee_notes") == demo + 1  # only shown

    maintenance.main(["purge-notes", "--days", "500", "--yes"], now=later)
    assert count(engine, "trainee_notes") == demo + 1  # within a longer period: kept

    maintenance.main(["purge-notes", "--yes"], now=later)
    assert "Deleted 1 notes, 1 talks and 1 tasks." in capsys.readouterr().out
    assert (count(engine, "trainee_notes"), count(engine, "personal_talks"), count(engine, "talk_proposals")) == (demo, 0, 0)
    assert count(engine, "tasks") == tasks_before - 1  # the task about him; every other task stays


def test_purge_with_nothing_to_delete_says_so(engine, academy, capsys):
    maintenance.main(["purge-notes", "--yes"], now=datetime.now(UTC))
    assert "0 notes, 0 talks and 0 tasks" in capsys.readouterr().out


@pytest.mark.parametrize("days", ["-1", "1000000"])
def test_an_impossible_period_is_refused(academy, days):
    with pytest.raises(SystemExit):
        maintenance.main(["purge-notes", "--days", days])


def test_a_trainee_row_is_locked_while_it_is_purged(engine, academy):
    """While a purge holds the row, a reopening request has to wait for it (SELECT ... FOR UPDATE)."""
    with repositories(engine) as repos:
        trainee = next(t for t in repos.trainees.list_all() if t.status is TraineeStatus.LEFT)
        repos.trainees.lock(trainee.key)
        with engine.connect() as other:
            other.execute(text("SET lock_timeout = '200ms'"))
            with pytest.raises(OperationalError, match="lock timeout"):
                other.execute(text(f"UPDATE trainees SET status = 'active', closed_at = NULL WHERE id = {trainee.key}"))
