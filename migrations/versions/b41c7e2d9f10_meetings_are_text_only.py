"""meetings are text only: no audio, so every meeting has a transcript

Revision ID: b41c7e2d9f10
Revises: e7d0780eaab2
Create Date: 2026-10-19 10:05:00.000000

Audio recordings were dropped (decision 2026-10-19). A meeting that was still waiting for a
recording ('uploaded') can never be processed, and has no text, proposals or tasks: it is deleted
(its participants go with it, ON DELETE CASCADE). Then the database itself guarantees the rule.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b41c7e2d9f10'
down_revision: Union[str, Sequence[str], None] = 'e7d0780eaab2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("DELETE FROM staff_meetings WHERE status = 'uploaded' AND transcript IS NULL")
    op.alter_column('staff_meetings', 'transcript', existing_type=sa.Text(), nullable=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column('staff_meetings', 'transcript', existing_type=sa.Text(), nullable=True)
