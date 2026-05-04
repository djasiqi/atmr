"""Add booking_messages table.

Mini-canal de communication entre entreprise et institution pour un booking.
Enum PostgreSQL bookingmessagesender (COMPANY, INSTITUTION).

Revision ID: 20260217_bmsg
Revises: 20260216_link_sugg
Create Date: 2026-02-17
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ENUM

revision = "20260217_bmsg"
down_revision = "20260216_link_sugg"
branch_labels = None
depends_on = None

sender_enum = ENUM(
    "COMPANY", "INSTITUTION", name="bookingmessagesender", create_type=False
)


def upgrade():
    sender_enum.create(op.get_bind(), checkfirst=True)

    op.create_table(
        "booking_messages",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "booking_id",
            sa.Integer(),
            sa.ForeignKey("booking.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "sender_user_id",
            sa.Integer(),
            sa.ForeignKey("user.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("sender_type", sender_enum, nullable=False),
        sa.Column("sender_label", sa.String(200), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )

    op.create_index(
        "ix_bmsg_booking_created",
        "booking_messages",
        ["booking_id", "created_at"],
    )


def downgrade():
    op.drop_index("ix_bmsg_booking_created", table_name="booking_messages")
    op.drop_table("booking_messages")
    sender_enum.drop(op.get_bind(), checkfirst=True)
