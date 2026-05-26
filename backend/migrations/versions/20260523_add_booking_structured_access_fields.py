"""Add structured pickup/dropoff access fields to booking.

Revision ID: 20260523_booking_access
Revises: 20260520_msg_idem_uq
Create Date: 2026-05-23 13:36:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "20260523_booking_access"
down_revision = "20260520_msg_idem_uq"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("pickup_floor", sa.String(length=50), nullable=True)
        )
        batch_op.add_column(
            sa.Column("pickup_door_code", sa.String(length=50), nullable=True)
        )
        batch_op.add_column(
            sa.Column("dropoff_floor", sa.String(length=50), nullable=True)
        )
        batch_op.add_column(
            sa.Column("dropoff_door_code", sa.String(length=50), nullable=True)
        )


def downgrade():
    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.drop_column("dropoff_door_code")
        batch_op.drop_column("dropoff_floor")
        batch_op.drop_column("pickup_door_code")
        batch_op.drop_column("pickup_floor")
