"""add_rl_suggestions_table

Revision ID: rl_suggestions_001
Revises: rl_feedback_001
Create Date: 2025-01-20

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "rl_suggestions_001"
down_revision = "44a28b52dc46"  # Pointe vers la migration la plus récente
branch_labels = None
depends_on = None


def upgrade():
    """Créer la table rl_suggestions pour le shadow mode."""
    op.create_table(
        "rl_suggestions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dispatch_run_id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("kpi_snapshot", sa.JSON(), nullable=True, comment="Snapshot des KPIs au moment de la suggestion"),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], name="fk_rl_suggestions_booking_id"),
        sa.ForeignKeyConstraint(["dispatch_run_id"], ["dispatch_run.id"], name="fk_rl_suggestions_dispatch_run_id"),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], name="fk_rl_suggestions_driver_id"),
        sa.PrimaryKeyConstraint("id")
    )

    # Créer index pour performance
    op.create_index("ix_rl_suggestions_dispatch_run_id", "rl_suggestions", ["dispatch_run_id"])
    op.create_index("ix_rl_suggestions_booking_id", "rl_suggestions", ["booking_id"])
    op.create_index("ix_rl_suggestions_driver_id", "rl_suggestions", ["driver_id"])
    op.create_index("ix_rl_suggestions_created_at", "rl_suggestions", ["created_at"])


def downgrade():
    """Supprimer la table rl_suggestions."""
    op.drop_index("ix_rl_suggestions_created_at", table_name="rl_suggestions")
    op.drop_index("ix_rl_suggestions_driver_id", table_name="rl_suggestions")
    op.drop_index("ix_rl_suggestions_booking_id", table_name="rl_suggestions")
    op.drop_index("ix_rl_suggestions_dispatch_run_id", table_name="rl_suggestions")
    op.drop_table("rl_suggestions")
