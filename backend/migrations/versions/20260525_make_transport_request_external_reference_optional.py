"""Make transport_requests.external_reference optional.

Revision ID: 20260525_req_ext_ref_optional
Revises: 20260523_booking_access
Create Date: 2026-05-25 14:46:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "20260525_req_ext_ref_optional"
down_revision = "20260523_booking_access"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("transport_requests", schema=None) as batch_op:
        batch_op.alter_column(
            "external_reference",
            existing_type=sa.String(length=100),
            nullable=True,
        )


def downgrade():
    with op.batch_alter_table("transport_requests", schema=None) as batch_op:
        batch_op.alter_column(
            "external_reference",
            existing_type=sa.String(length=100),
            nullable=False,
        )
