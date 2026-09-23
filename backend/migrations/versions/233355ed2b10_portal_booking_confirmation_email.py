"""portal_booking_confirmation_email

Revision ID: 233355ed2b10
Revises: 96428b10a902
Create Date: 2026-09-23 20:01:05.867395

Table générée par ``flask db revision --autogenerate``.
Les écarts d'index sans rapport ont été retirés.
Aucune donnée n'est insérée.
"""

from alembic import op
import sqlalchemy as sa


revision = "233355ed2b10"
down_revision = "96428b10a902"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "portal_booking_confirmation_email",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("contract_event_id", sa.Integer(), nullable=True),
        sa.Column("recipient_email", sa.String(length=255), nullable=True),
        sa.Column("template_version", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('sent', 'failed')",
            name="ck_portal_booking_confirmation_email_status",
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["contract_event_id"],
            ["client_booking_contract_event.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_portal_booking_confirmation_email_booking_id",
        "portal_booking_confirmation_email",
        ["booking_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_portal_booking_confirmation_email_booking_id",
        table_name="portal_booking_confirmation_email",
    )
    op.drop_table("portal_booking_confirmation_email")
