"""✅ B2: Add decision_explanation to assignment table.

Revision ID: add_decision_explanation
Revises: d7e8f9a1b2c3
Create Date: 2025-01-27 15:00:00.000000

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = "add_decision_explanation"
down_revision = "d7e8f9a1b2c3"
branch_labels = None
depends_on = None


def upgrade():
    # ✅ B2: Ajouter decision_explanation (JSONB) à assignment
    op.add_column(
        "assignment",
        sa.Column(
            "decision_explanation",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True
        )
    )


def downgrade():
    op.drop_column("assignment", "decision_explanation")

