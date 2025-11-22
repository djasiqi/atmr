"""add_rl_suggestion_metrics_table

Revision ID: rl_metrics_001
Revises:
Create Date: 2025-10-21

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "rl_metrics_001"
down_revision = "abc123456789"  # Pointe vers add_autonomous_action_table
branch_labels = None
depends_on = None


def upgrade():
    """Créer la table rl_suggestion_metrics pour tracker les métriques RL."""
    op.create_table(
        "rl_suggestion_metrics",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("suggestion_id", sa.String(length=100), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("assignment_id", sa.Integer(), nullable=False),
        sa.Column("current_driver_id", sa.Integer(), nullable=False),
        sa.Column("suggested_driver_id", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("expected_gain_minutes", sa.Integer(), server_default="0"),
        sa.Column("q_value", sa.Float(), nullable=True),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column(
            "generated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("applied_at", sa.DateTime(), nullable=True),
        sa.Column("rejected_at", sa.DateTime(), nullable=True),
        sa.Column("actual_gain_minutes", sa.Integer(), nullable=True),
        sa.Column("was_successful", sa.Boolean(), nullable=True),
        sa.Column("additional_data", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("suggestion_id"),
    )

    # Créer index pour performance
    op.create_index(
        "ix_rl_suggestion_metrics_company_id", "rl_suggestion_metrics", ["company_id"]
    )
    op.create_index(
        "ix_rl_suggestion_metrics_suggestion_id",
        "rl_suggestion_metrics",
        ["suggestion_id"],
    )
    op.create_index(
        "ix_rl_suggestion_metrics_booking_id", "rl_suggestion_metrics", ["booking_id"]
    )
    op.create_index(
        "ix_rl_suggestion_metrics_generated_at",
        "rl_suggestion_metrics",
        ["generated_at"],
    )


def downgrade():
    """Supprimer la table rl_suggestion_metrics."""
    op.drop_index(
        "ix_rl_suggestion_metrics_generated_at", table_name="rl_suggestion_metrics"
    )
    op.drop_index(
        "ix_rl_suggestion_metrics_booking_id", table_name="rl_suggestion_metrics"
    )
    op.drop_index(
        "ix_rl_suggestion_metrics_suggestion_id", table_name="rl_suggestion_metrics"
    )
    op.drop_index(
        "ix_rl_suggestion_metrics_company_id", table_name="rl_suggestion_metrics"
    )
    op.drop_table("rl_suggestion_metrics")
