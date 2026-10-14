"""Database connection: one engine per process, and a unit-of-work helper."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from pacer.config import get_settings


@lru_cache
def get_engine() -> Engine:
    """The connection pool to PostgreSQL, created once and reused."""
    return create_engine(get_settings().database_url)


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
