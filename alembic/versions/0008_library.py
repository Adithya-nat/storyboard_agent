"""film library: films creators publish to the homepage

Revision ID: 0008_library
Revises: 0007_run_controls
"""
import sqlalchemy as sa
from alembic import op

revision = "0008_library"
down_revision = "0007_run_controls"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "library_films",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("job_id", sa.String(36), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="rendering"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("title", sa.String(120), nullable=False),
        sa.Column("native_title", sa.String(160), nullable=True),
        sa.Column("tagline", sa.String(160), nullable=True),
        sa.Column("logline", sa.Text(), nullable=False, server_default=""),
        sa.Column("genre", sa.String(48), nullable=False, server_default=""),
        sa.Column("prompt", sa.Text(), nullable=True),
        sa.Column("language", sa.String(8), nullable=False, server_default="en"),
        sa.Column("style", sa.String(16), nullable=False, server_default="dialogue"),
        sa.Column("length_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("scenes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("made_minutes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0"),
        sa.Column("poster_look", sa.String(24), nullable=False, server_default="auto"),
        sa.Column("poster_frame", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("poster_version", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("posters_made", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_library_films_job_id", "library_films", ["job_id"])
    op.create_index("ix_library_films_user_id", "library_films", ["user_id"])
    op.create_index("ix_library_films_published_at", "library_films", ["published_at"])


def downgrade() -> None:
    op.drop_index("ix_library_films_published_at", "library_films")
    op.drop_index("ix_library_films_user_id", "library_films")
    op.drop_index("ix_library_films_job_id", "library_films")
    op.drop_table("library_films")
