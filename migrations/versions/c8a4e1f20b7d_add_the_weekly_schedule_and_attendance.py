"""add the weekly schedule and attendance

Revision ID: c8a4e1f20b7d
Revises: b41c7e2d9f10
Create Date: 2026-10-26 19:00:00.000000

Three tables (decision 2026-10-26):
* schedule_slots      the template of a typical week (weekday, times, title; a group or the cohort)
* scheduled_sessions  one slot on one date. Publishing a week creates them from the template; the
                      unique (slot_id, session_date) makes publishing the same week twice add nothing
* attendance          one row per (session, trainee): present, absent or excused. No reason text,
                      on purpose (health data is never written). Deleting a session deletes its rows.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c8a4e1f20b7d'
down_revision: Union[str, Sequence[str], None] = 'b41c7e2d9f10'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        'schedule_slots',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('cohort_id', sa.Integer(), nullable=False),
        sa.Column('group_id', sa.Integer(), nullable=True),
        sa.Column('weekday', sa.Integer(), nullable=False),
        sa.Column('start_time', sa.Time(), nullable=False),
        sa.Column('end_time', sa.Time(), nullable=False),
        sa.Column('title', sa.String(length=100), nullable=False),
        sa.Column('active', sa.Boolean(), nullable=False),
        sa.CheckConstraint('end_time > start_time', name='ck_schedule_slots_times'),
        sa.ForeignKeyConstraint(['cohort_id'], ['cohorts.id']),
        sa.ForeignKeyConstraint(['group_id'], ['groups.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(op.f('ix_schedule_slots_cohort_id'), 'schedule_slots', ['cohort_id'], unique=False)
    op.create_table(
        'scheduled_sessions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('cohort_id', sa.Integer(), nullable=False),
        sa.Column('group_id', sa.Integer(), nullable=True),
        sa.Column('session_date', sa.Date(), nullable=False),
        sa.Column('start_time', sa.Time(), nullable=False),
        sa.Column('end_time', sa.Time(), nullable=False),
        sa.Column('title', sa.String(length=100), nullable=False),
        sa.Column('slot_id', sa.Integer(), nullable=True),
        sa.Column('cancelled', sa.Boolean(), nullable=False),
        sa.CheckConstraint('end_time > start_time', name='ck_scheduled_sessions_times'),
        sa.ForeignKeyConstraint(['cohort_id'], ['cohorts.id']),
        sa.ForeignKeyConstraint(['group_id'], ['groups.id']),
        sa.ForeignKeyConstraint(['slot_id'], ['schedule_slots.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('slot_id', 'session_date', name='uq_scheduled_sessions_slot_date'),
    )
    op.create_index(
        op.f('ix_scheduled_sessions_session_date'), 'scheduled_sessions', ['session_date'], unique=False
    )
    op.create_table(
        'attendance',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('session_id', sa.Integer(), nullable=False),
        sa.Column('trainee_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=10), nullable=False),
        sa.Column('recorded_by_id', sa.Integer(), nullable=False),
        sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("status IN ('present', 'absent', 'excused')", name='ck_attendance_status'),
        sa.ForeignKeyConstraint(['recorded_by_id'], ['staff_members.id']),
        sa.ForeignKeyConstraint(['session_id'], ['scheduled_sessions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['trainee_id'], ['trainees.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id', 'trainee_id', name='uq_attendance_session_trainee'),
    )
    op.create_index(op.f('ix_attendance_trainee_id'), 'attendance', ['trainee_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_attendance_trainee_id'), table_name='attendance')
    op.drop_table('attendance')
    op.drop_index(op.f('ix_scheduled_sessions_session_date'), table_name='scheduled_sessions')
    op.drop_table('scheduled_sessions')
    op.drop_index(op.f('ix_schedule_slots_cohort_id'), table_name='schedule_slots')
    op.drop_table('schedule_slots')
