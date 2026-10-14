"""Database connection: one engine per process, and a unit-of-work helper."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from pacer.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """The connection pool to PostgreSQL, created once and reused.

    timezone=UTC: PostgreSQL returns TIMESTAMP WITH TIME ZONE values in the session's
    time zone. Pinning it to UTC means every time leaves the database in UTC, whatever
    the server's local zone is (the same instant either way, but one consistent form).
    """
    settings = get_settings()
    return create_engine(
        settings.database_url,
        connect_args={"options": "-c timezone=UTC"},
        echo=settings.sql_echo,  # SQL_ECHO=true: log every statement SQLAlchemy sends
        json_serializer=_json_dumps,
    )


def _json_dumps(value: object) -> str:
    """JSON columns keep Hebrew as Hebrew. The default (ensure_ascii) stores it as \\u05d0... escapes."""
    return json.dumps(value, ensure_ascii=False)


@contextmanager
def unit_of_work(engine: Engine | None = None) -> Iterator[Session]:
    """One transaction: commits if the block succeeds, rolls back if it raises.

    Usage:
        with unit_of_work() as session:
            trainees = TraineeRepository(session)
            goals = GoalRepository(session)
            ...  # every change here is saved together, or not at all
    """
    with Session(engine or get_engine(), expire_on_commit=False) as session:
        with session.begin():
            yield session
