"""job_stages + job_logs tables (per-stage status + streamable logs)

Revision ID: 0002_stages_logs
Revises: 0001_initial
Create Date: 2026-06-27

job_logs.id is a plain Integer PK -> SERIAL/IDENTITY on Postgres (and INTEGER
PRIMARY KEY rowid-alias if ever run on SQLite). It is the streaming cursor
(SSE Last-Event-ID + "logs after X"); the (job_id, id) index serves the
incremental fetch ``WHERE job_id=? AND id>?``.
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_stages_logs"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job_stages",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("name", sa.String(length=32), nullable=False),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("progress_current", sa.Integer(), nullable=True),
        sa.Column("progress_total", sa.Integer(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("job_id", "name", name="uq_job_stages_job_name"),
    )
    op.create_index("ix_job_stages_job_id", "job_stages", ["job_id"])

    op.create_table(
        "job_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("job_id", sa.String(length=36), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=True),
        sa.Column("level", sa.String(length=8), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_job_logs_job_id_id", "job_logs", ["job_id", "id"])


def downgrade() -> None:
    op.drop_index("ix_job_logs_job_id_id", table_name="job_logs")
    op.drop_table("job_logs")
    op.drop_index("ix_job_stages_job_id", table_name="job_stages")
    op.drop_table("job_stages")
