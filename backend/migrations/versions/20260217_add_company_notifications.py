"""Add company_notifications table.

Revision ID: 20260217_comp_notif
Revises: 20260217_add_booking_messages
Create Date: 2026-02-17
"""

import sqlalchemy as sa
from alembic import op

revision = "20260217_comp_notif"
down_revision = "20260217_bmsg"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "company_notifications",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "company_id",
            sa.Integer(),
            sa.ForeignKey("company.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(50), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column("is_read", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("dedupe_key", sa.String(200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("company_id", "dedupe_key", name="uq_comp_notif_dedupe"),
    )
    op.create_index(
        "ix_comp_notif_company_read_created",
        "company_notifications",
        ["company_id", "is_read", sa.text("created_at DESC")],
    )
    op.create_index(
        "ix_comp_notif_company_created",
        "company_notifications",
        ["company_id", sa.text("created_at DESC")],
    )


def downgrade():
    op.drop_index("ix_comp_notif_company_created", table_name="company_notifications")
    op.drop_index("ix_comp_notif_company_read_created", table_name="company_notifications")
    op.drop_table("company_notifications")
