"""user_id on jobs, spend_events and run_ledger (hosted multi-user mode)

Revision ID: 0006_users_byok
Revises: 0005_film_score
"""
import sqlalchemy as sa
from alembic import op

revision = "0006_users_byok"
down_revision = "0005_film_score"
branch_labels = None
depends_on = None

TABLES = ("jobs", "spend_events", "run_ledger")


def upgrade() -> None:
    for table in TABLES:
        op.add_column(table, sa.Column("user_id", sa.String(64), nullable=True))
        op.create_index(f"ix_{table}_user_id", table, ["user_id"])


def downgrade() -> None:
    for table in TABLES:
        op.drop_index(f"ix_{table}_user_id", table_name=table)
        op.drop_column(table, "user_id")
