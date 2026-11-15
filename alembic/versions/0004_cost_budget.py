"""cost tracking + persistent spend history + story style/language

Revision ID: 0004_cost_budget
Revises: 0003_token_usage
"""
import sqlalchemy as sa
from alembic import op

revision = "0004_cost_budget"
down_revision = "0003_token_usage"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("jobs", sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"))
    op.add_column("jobs", sa.Column("style", sa.String(16), nullable=False, server_default="narrated"))
    op.add_column("jobs", sa.Column("language", sa.String(8), nullable=False, server_default="en"))
    op.add_column("job_stages", sa.Column("cost_usd", sa.Float(), nullable=True))
    op.create_table(
        "spend_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("stage", sa.String(32), nullable=True),
        sa.Column("category", sa.String(16), nullable=False),
        sa.Column("provider", sa.String(16), nullable=False, server_default="google"),
        sa.Column("model", sa.String(64), nullable=False),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("units", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_spend_events_job_id", "spend_events", ["job_id"])
    op.create_index("ix_spend_events_created_at", "spend_events", ["created_at"])
    op.create_table(
        "run_ledger",
        sa.Column("job_id", sa.String(36), primary_key=True),
        sa.Column("prompt", sa.Text(), nullable=False, server_default=""),
        sa.Column("style", sa.String(16), nullable=False, server_default="narrated"),
        sa.Column("language", sa.String(8), nullable=False, server_default="en"),
        sa.Column("duration", sa.Integer(), nullable=False, server_default="30"),
        sa.Column("status", sa.String(32), nullable=False, server_default="running"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("cost_by_category", sa.JSON(), nullable=False),
        sa.Column("total_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("video_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("images", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("scenes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("omni_scenes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("fallback_scenes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("judge_score", sa.Integer(), nullable=True),
        sa.Column("runtime_seconds", sa.Float(), nullable=False, server_default="0"),
        sa.Column("models", sa.JSON(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_run_ledger_started_at", "run_ledger", ["started_at"])


def downgrade() -> None:
    op.drop_index("ix_run_ledger_started_at", "run_ledger")
    op.drop_table("run_ledger")
    op.drop_index("ix_spend_events_created_at", "spend_events")
    op.drop_index("ix_spend_events_job_id", "spend_events")
    op.drop_table("spend_events")
    op.drop_column("job_stages", "cost_usd")
    op.drop_column("jobs", "language")
    op.drop_column("jobs", "style")
    op.drop_column("jobs", "cost_usd")
