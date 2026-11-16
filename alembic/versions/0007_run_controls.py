"""run controls: pause flag + breakpoints on jobs

Revision ID: 0007_run_controls
Revises: 0006_users_byok
"""
import sqlalchemy as sa
from alembic import op

revision = "0007_run_controls"
down_revision = "0006_users_byok"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("pause_requested_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("jobs", sa.Column("breakpoints", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("jobs", "breakpoints")
    op.drop_column("jobs", "pause_requested_at")
