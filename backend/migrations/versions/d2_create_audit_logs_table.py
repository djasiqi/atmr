"""✅ D2: Create audit_logs table for append-only audit trail.

Revision ID: d2_audit_logs
Revises: add_decision_explanation
Create Date: 2025-01-28 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "d2_audit_logs"
down_revision = "add_decision_explanation"
branch_labels = None
depends_on = None


def upgrade():
    # ✅ D2: Table d'audit append-only (pas de UPDATE/DELETE)
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("user_type", sa.String(length=50), nullable=False),
        sa.Column("action_type", sa.String(length=100), nullable=False),
        sa.Column("action_category", sa.String(length=50), nullable=False),
        sa.Column("action_details", sa.Text(), nullable=False),
        sa.Column("result_status", sa.String(length=50), nullable=False),
        sa.Column("result_message", sa.Text(), nullable=True),
        sa.Column("company_id", sa.Integer(), nullable=True),
        sa.Column("booking_id", sa.Integer(), nullable=True),
        sa.Column("driver_id", sa.Integer(), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("metadata", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )

    # Indexes pour performance de requêtes d'audit
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])
    op.create_index("ix_audit_logs_user_id", "audit_logs", ["user_id"])
    op.create_index("ix_audit_logs_action_type", "audit_logs", ["action_type"])
    op.create_index("ix_audit_logs_action_category", "audit_logs", ["action_category"])
    op.create_index("ix_audit_logs_company_id", "audit_logs", ["company_id"])
    op.create_index("ix_audit_logs_booking_id", "audit_logs", ["booking_id"])
    op.create_index("ix_audit_logs_driver_id", "audit_logs", ["driver_id"])


def downgrade():
    # Supprimer les indexes
    op.drop_index("ix_audit_logs_driver_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_booking_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_company_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_action_category", table_name="audit_logs")
    op.drop_index("ix_audit_logs_action_type", table_name="audit_logs")
    op.drop_index("ix_audit_logs_user_id", table_name="audit_logs")
    op.drop_index("ix_audit_logs_created_at", table_name="audit_logs")

    # Supprimer la table
    op.drop_table("audit_logs")
