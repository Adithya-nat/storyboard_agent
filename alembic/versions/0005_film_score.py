"""film_score on the run ledger (final-film coherence review)

Revision ID: 0005_film_score
Revises: 0004_cost_budget
"""
import sqlalchemy as sa
from alembic import op

revision = "0005_film_score"
down_revision = "0004_cost_budget"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("run_ledger", sa.Column("film_score", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("run_ledger", "film_score")
