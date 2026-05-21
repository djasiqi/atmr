"""Add message hub fields (thread, type, priority, ack).

Revision ID: 20260519_msg_hub
Revises:
Create Date: 2026-05-19
"""

from alembic import op
import sqlalchemy as sa

revision = "20260519_msg_hub"
down_revision = "20260428_line_meta"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("message", schema=None) as batch_op:
        batch_op.add_column(sa.Column("thread_id", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("booking_id", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column("message_type", sa.String(length=32), nullable=False, server_default="text")
        )
        batch_op.add_column(
            sa.Column("priority", sa.String(length=16), nullable=False, server_default="normal")
        )
        batch_op.add_column(sa.Column("client_message_id", sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column("acked_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.create_index("ix_message_thread_id", ["thread_id"], unique=False)
        batch_op.create_index("ix_message_booking_id", ["booking_id"], unique=False)
        batch_op.create_index("ix_message_client_message_id", ["client_message_id"], unique=False)


def downgrade():
    with op.batch_alter_table("message", schema=None) as batch_op:
        batch_op.drop_index("ix_message_client_message_id")
        batch_op.drop_index("ix_message_booking_id")
        batch_op.drop_index("ix_message_thread_id")
        batch_op.drop_column("acked_at")
        batch_op.drop_column("client_message_id")
        batch_op.drop_column("priority")
        batch_op.drop_column("message_type")
        batch_op.drop_column("booking_id")
        batch_op.drop_column("thread_id")
