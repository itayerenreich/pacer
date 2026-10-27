"""The demo loader against PostgreSQL: loads, refuses to overwrite, resets, and the API serves it."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from pacer.api.app import create_app
from pacer.api.dependencies import get_db_engine
from pacer.demo.__main__ import load
from pacer.repositories import unit_of_work
from pacer.repositories.tables import GoalUpdateRow, StaffMemberRow, TraineeNoteRow, TraineeRow

pytestmark = pytest.mark.integration


def count(engine, table) -> int:
    with unit_of_work(engine) as session:
        return session.scalar(select(func.count()).select_from(table))


def test_load_fills_the_database(engine):
    summary = load(engine, reset=False, seed=2026)

    assert count(engine, TraineeRow) == 35
    assert count(engine, StaffMemberRow) == 5
    assert count(engine, GoalUpdateRow) == summary.update_count > 0
    assert count(engine, TraineeNoteRow) == summary.note_count > 0


def test_refuses_to_load_twice_without_reset(engine):
    load(engine, reset=False, seed=2026)
    with pytest.raises(SystemExit, match="already has data"):
        load(engine, reset=False, seed=2026)
    assert count(engine, TraineeRow) == 35  # nothing was added


def test_reset_replaces_the_data_and_restarts_ids(engine):
    first = load(engine, reset=False, seed=2026)
    second = load(engine, reset=True, seed=2026)

    assert count(engine, TraineeRow) == 35
    assert second.manager.id == first.manager.id == 1  # RESTART IDENTITY: same ids every time


def test_the_api_serves_the_demo_academy(engine):
    summary = load(engine, reset=False, seed=2026)
    app = create_app()
    app.dependency_overrides[get_db_engine] = lambda: engine
    manager = {"X-Staff-Id": str(summary.manager.id)}

    with TestClient(app) as client:
        pending = client.get("/approvals/pending", headers=manager).json()
        trainee = client.get(f"/trainees/{summary.trainees[0].id}", headers=manager).json()
        everyone = client.get("/trainees", headers=manager).json()
        active = client.get("/trainees?status=active", headers=manager).json()
        noa = client.get("/trainees", headers={"X-Staff-Id": str(summary.counselors[0].id)}).json()

    assert len(pending) == 2
    assert (len(everyone), len(active)) == (35, 33)
    assert 0 < len(noa) < 35 and {t["group_id"] for t in noa} == {summary.groups[0].id}  # her group only
    assert trainee["first_name"] == summary.trainees[0].first_name  # Hebrew survives the round trip


def test_reset_keeps_the_library(engine):
    """The library is imported from the academy's files and indexed for minutes: not demo data. Found
    2026-10-27, when a reset emptied it and every library question got "no source"."""
    from pacer.domain import Chunk, EvidenceLevel, KnowledgeSource, SourceKind, SourceLicense
    from pacer.repositories import repositories

    load(engine, reset=False, seed=2026)
    with repositories(engine) as repos:
        source = repos.sources.add(
            KnowledgeSource(
                title="איגרת הרמב\"ן",
                kind=SourceKind.CLASSIC,
                evidence_level=EvidenceLevel.TRADITION,
                license=SourceLicense.PUBLIC_DOMAIN,
            )
        )
        repos.chunks.add_many([Chunk(source_id=source.key, chunking="fixed-200-40", ordinal=0, text="תתנהג תמיד לדבר כל דבריך בנחת")])
    load(engine, reset=True, seed=2026)
    with repositories(engine) as repos:
        assert [s.title for s in repos.sources.list_all()] == ["איגרת הרמב\"ן"]
    assert count(engine, TraineeRow) == 35  # the academy itself was replaced
