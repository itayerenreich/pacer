"""SQLAlchemy table definitions: how domain objects are stored in PostgreSQL.

These classes describe tables, not business rules. The rules live in the
domain layer; repositories translate between domain models and these rows.
"""

from datetime import date, datetime

from sqlalchemy import JSON, Column, DateTime, ForeignKey, String, Table, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Parent of every table. Collects all table definitions in Base.metadata."""


# Many-to-many link: which counselors guide which group.
group_counselors = Table(
    "group_counselors",
    Base.metadata,
    Column("group_id", ForeignKey("groups.id", ondelete="CASCADE"), primary_key=True),
    Column("counselor_id", ForeignKey("staff_members.id", ondelete="CASCADE"), primary_key=True),
)


class CohortRow(Base):
    __tablename__ = "cohorts"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    start_date: Mapped[date]
    end_date: Mapped[date]


class StaffMemberRow(Base):
    __tablename__ = "staff_members"

    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(100))
    role: Mapped[str] = mapped_column(String(20))
    responsibility_area: Mapped[str | None] = mapped_column(String(30))


class GroupRow(Base):
    __tablename__ = "groups"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50))
    cohort_id: Mapped[int] = mapped_column(ForeignKey("cohorts.id"), index=True)


class TraineeRow(Base):
    __tablename__ = "trainees"

    id: Mapped[int] = mapped_column(primary_key=True)
    first_name: Mapped[str] = mapped_column(String(50))
    last_name: Mapped[str] = mapped_column(String(50))
    cohort_id: Mapped[int] = mapped_column(ForeignKey("cohorts.id"), index=True)
    group_id: Mapped[int] = mapped_column(ForeignKey("groups.id"), index=True)
    primary_counselor_id: Mapped[int] = mapped_column(ForeignKey("staff_members.id"), index=True)
    status: Mapped[str] = mapped_column(String(20))


class PersonalFileRow(Base):
    __tablename__ = "personal_files"

    id: Mapped[int] = mapped_column(primary_key=True)
    trainee_id: Mapped[int] = mapped_column(ForeignKey("trainees.id"), unique=True)
    intake_answers: Mapped[dict[str, str]] = mapped_column(JSON, default=dict)
    background_notes: Mapped[str | None] = mapped_column(Text)


class GoalRow(Base):
    __tablename__ = "goals"

    id: Mapped[int] = mapped_column(primary_key=True)
    trainee_id: Mapped[int] = mapped_column(ForeignKey("trainees.id"), index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(30))
    status: Mapped[str] = mapped_column(String(20))
    progress: Mapped[int]
    target_date: Mapped[date | None]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class GoalUpdateRow(Base):
    __tablename__ = "goal_updates"

    id: Mapped[int] = mapped_column(primary_key=True)
    goal_id: Mapped[int] = mapped_column(ForeignKey("goals.id"), index=True)
    author_id: Mapped[int] = mapped_column(ForeignKey("staff_members.id"))
    progress: Mapped[int]
    note: Mapped[str] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
