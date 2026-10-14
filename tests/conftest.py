"""Shared pytest fixtures.

Integration tests run against a real PostgreSQL in a throwaway Docker container
(testcontainers). The container starts once per test run, the Alembic migrations
build the tables, and the container is deleted when pytest finishes.

Isolation between tests: after every test, all tables are emptied with TRUNCATE.
This is simple, and the code under test commits exactly as it does in production.

When the suite grows (hundreds of DB tests) and TRUNCATE becomes the bottleneck,
switch to the faster "rollback per test" strategy:
  1. Let unit_of_work() accept an Engine OR a Connection (Session(bind=...) takes both).
  2. In the `engine` fixture: open a connection, begin an outer transaction, and give
     the tests that connection instead of the engine.
  3. Create sessions with join_transaction_mode="create_savepoint", so session.commit()
     only releases a SAVEPOINT inside the outer transaction.
  4. After the test, roll back the outer transaction: every change disappears, no TRUNCATE.
See the "test isolation" row in the architecture decision log (project plan).
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from testcontainers.community.postgres import PostgresContainer

from pacer.repositories.tables import Base

POSTGRES_IMAGE = "postgres:17"  # same version as compose.yaml
ALEMBIC_INI = Path(__file__).resolve().parents[1] / "alembic.ini"


@pytest.fixture(scope="session")
def migrated_engine() -> Iterator[Engine]:
    """One PostgreSQL container for the whole test run, with all migrations applied."""
    with PostgresContainer(POSTGRES_IMAGE, driver="psycopg") as postgres:
        engine = create_engine(postgres.get_connection_url())

        alembic_config = Config(str(ALEMBIC_INI))
        with engine.begin() as connection:
            # migrations/env.py uses this connection instead of the database in .env
            alembic_config.attributes["connection"] = connection
            command.upgrade(alembic_config, "head")

        yield engine
        engine.dispose()


@pytest.fixture
def engine(migrated_engine: Engine) -> Iterator[Engine]:
    """The test database engine. Every table is emptied after each test."""
    yield migrated_engine

    table_names = ", ".join(table.name for table in Base.metadata.sorted_tables)
    with migrated_engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {table_names} RESTART IDENTITY CASCADE"))
