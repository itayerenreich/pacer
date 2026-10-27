"""link a task to a trainee

Revision ID: a3f9c2d17e54
Revises: c8a4e1f20b7d
Create Date: 2026-10-27 15:30:00.000000

Tasks can now be written by hand (decision 2026-10-27), and a task may be about a trainee ("talk to
Nadav by Thursday"): it then shows in the trainee's profile too. Nullable: most tasks are about no one.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a3f9c2d17e54'
down_revision: Union[str, Sequence[str], None] = 'c8a4e1f20b7d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('tasks', sa.Column('trainee_id', sa.Integer(), nullable=True))
    op.create_index(op.f('ix_tasks_trainee_id'), 'tasks', ['trainee_id'], unique=False)
    op.create_foreign_key(None, 'tasks', 'trainees', ['trainee_id'], ['id'])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint('tasks_trainee_id_fkey', 'tasks', type_='foreignkey')  # PostgreSQL's default name
    op.drop_index(op.f('ix_tasks_trainee_id'), table_name='tasks')
    op.drop_column('tasks', 'trainee_id')
