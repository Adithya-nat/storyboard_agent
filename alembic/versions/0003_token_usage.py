"""token accounting columns (jobs totals + per-stage breakdown)

Revision ID: 0003_token_usage
Revises: 0002_stages_logs
Create Date: 2026-06-28

Adds whole-job token totals to ``jobs`` and a per-stage total to ``job_stages``.
``server_default="0"`` keeps the NOT NULL job columns valid for pre-existing rows on both
SQLite (add-column requires a default) and Postgres. ``job_stages.tokens`` is nullable
(a stage that ran before this column existed simply has no number).
"""
from alembic import op
import sqlalchemy as sa

revision = "0003_token_usage"
down_revision = "0002_stages_logs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("jobs", sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("jobs", sa.Column("output_tokens", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("job_stages", sa.Column("tokens", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("job_stages", "tokens")
    op.drop_column("jobs", "output_tokens")
    op.drop_column("jobs", "prompt_tokens")
    op.drop_column("jobs", "total_tokens")
