"""add_rl_feedbacks_table

Revision ID: rl_feedback_001
Revises: rl_metrics_001
Create Date: 2025-10-21

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "rl_feedback_001"
down_revision = "rl_metrics_001"  # Pointe vers add_rl_suggestion_metrics_table
branch_labels = None
depends_on = None


def upgrade():
    """Créer la table rl_feedbacks pour le feedback loop."""
    op.create_table(
        "rl_feedbacks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("suggestion_id", sa.String(length=100), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("assignment_id", sa.Integer(), nullable=False),
        sa.Column("current_driver_id", sa.Integer(), nullable=False),
        sa.Column("suggested_driver_id", sa.Integer(), nullable=False),
        sa.Column("action", sa.String(length=20), nullable=False),
        sa.Column("feedback_reason", sa.Text(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("suggestion_generated_at", sa.DateTime(), nullable=True),
        sa.Column("actual_outcome", sa.JSON(), nullable=True),
        sa.Column("was_successful", sa.Boolean(), nullable=True),
        sa.Column("actual_gain_minutes", sa.Integer(), nullable=True),
        sa.Column("suggestion_state", sa.JSON(), nullable=True),
        sa.Column("suggestion_action", sa.Integer(), nullable=True),
        sa.Column("suggestion_confidence", sa.Float(), nullable=True),
        sa.Column("additional_data", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Créer index pour performance
    op.create_index("ix_rl_feedbacks_company_id", "rl_feedbacks", ["company_id"])
    op.create_index("ix_rl_feedbacks_suggestion_id", "rl_feedbacks", ["suggestion_id"])
    op.create_index("ix_rl_feedbacks_booking_id", "rl_feedbacks", ["booking_id"])
    op.create_index("ix_rl_feedbacks_action", "rl_feedbacks", ["action"])
    op.create_index("ix_rl_feedbacks_created_at", "rl_feedbacks", ["created_at"])


def downgrade():
    """Supprimer la table rl_feedbacks."""
    op.drop_index("ix_rl_feedbacks_created_at", table_name="rl_feedbacks")
    op.drop_index("ix_rl_feedbacks_action", table_name="rl_feedbacks")
    op.drop_index("ix_rl_feedbacks_booking_id", table_name="rl_feedbacks")
    op.drop_index("ix_rl_feedbacks_suggestion_id", table_name="rl_feedbacks")
    op.drop_index("ix_rl_feedbacks_company_id", table_name="rl_feedbacks")
    op.drop_table("rl_feedbacks")
