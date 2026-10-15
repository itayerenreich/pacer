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


class ApprovalRequestRow(Base):
    __tablename__ = "approval_requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    kind: Mapped[str] = mapped_column(String(30))
    trainee_id: Mapped[int] = mapped_column(ForeignKey("trainees.id"), index=True)
    requested_by_id: Mapped[int] = mapped_column(ForeignKey("staff_members.id"), index=True)
    reason: Mapped[str] = mapped_column(String(1000))
    requested_status: Mapped[str | None] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20), index=True)
    decided_by_id: Mapped[int | None] = mapped_column(ForeignKey("staff_members.id"))
    decision_note: Mapped[str | None] = mapped_column(String(1000))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# Many-to-many link: which staff members took part in which meeting (same pattern as group_counselors).
meeting_participants = Table(
    "meeting_participants",
    Base.metadata,
    Column("meeting_id", ForeignKey("staff_meetings.id", ondelete="CASCADE"), primary_key=True),
    Column("staff_id", ForeignKey("staff_members.id", ondelete="CASCADE"), primary_key=True),
)


class StaffMeetingRow(Base):
    __tablename__ = "staff_meetings"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    held_on: Mapped[date]
    created_by_id: Mapped[int] = mapped_column(ForeignKey("staff_members.id"))
    status: Mapped[str] = mapped_column(String(30), index=True)
    transcript: Mapped[str | None] = mapped_column(Text)
    anonymized_transcript: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class ProposedTaskRow(Base):
    __tablename__ = "proposed_tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(ForeignKey("staff_meetings.id"), index=True)
    description: Mapped[str] = mapped_column(String(300))
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("staff_members.id"))
    due_date: Mapped[date | None]
    status: Mapped[str] = mapped_column(String(20))
    decided_by_id: Mapped[int | None] = mapped_column(ForeignKey("staff_members.id"))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    task_id: Mapped[int | None] = mapped_column(ForeignKey("tasks.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TaskRow(Base):
    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    description: Mapped[str] = mapped_column(String(300))
    owner_id: Mapped[int] = mapped_column(ForeignKey("staff_members.id"), index=True)
    due_date: Mapped[date | None]
    status: Mapped[str] = mapped_column(String(20))
    created_by_id: Mapped[int] = mapped_column(ForeignKey("staff_members.id"))
    source_meeting_id: Mapped[int | None] = mapped_column(ForeignKey("staff_meetings.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class PersonalTalkRow(Base):
    __tablename__ = "personal_talks"

    id: Mapped[int] = mapped_column(primary_key=True)
    trainee_id: Mapped[int] = mapped_column(ForeignKey("trainees.id"), index=True)
    counselor_id: Mapped[int] = mapped_column(ForeignKey("staff_members.id"))
    held_on: Mapped[date]
    status: Mapped[str] = mapped_column(String(20))
    notes: Mapped[str] = mapped_column(Text)
    anonymized_notes: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TraineeNoteRow(Base):
    __tablename__ = "trainee_notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    trainee_id: Mapped[int] = mapped_column(ForeignKey("trainees.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    text: Mapped[str] = mapped_column(String(300))
    author_id: Mapped[int] = mapped_column(ForeignKey("staff_members.id"))
    # SET NULL: deleting a talk keeps the notes a counselor approved from it
    source_talk_id: Mapped[int | None] = mapped_column(ForeignKey("personal_talks.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class TalkProposalRow(Base):
    __tablename__ = "talk_proposals"

    id: Mapped[int] = mapped_column(primary_key=True)
    talk_id: Mapped[int] = mapped_column(ForeignKey("personal_talks.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    goal_id: Mapped[int | None] = mapped_column(ForeignKey("goals.id"))
    progress: Mapped[int | None]
    note_kind: Mapped[str | None] = mapped_column(String(20))
    text: Mapped[str | None] = mapped_column(String(500))  # None once decided (data minimization)
    status: Mapped[str] = mapped_column(String(20))
    decided_by_id: Mapped[int | None] = mapped_column(ForeignKey("staff_members.id"))
    decided_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    goal_update_id: Mapped[int | None] = mapped_column(ForeignKey("goal_updates.id"))
    note_id: Mapped[int | None] = mapped_column(ForeignKey("trainee_notes.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
