"""add external_carrier_email

Revision ID: 97c225be0651
Revises: 5f8a87e796bb
Create Date: 2026-06-13 14:22:43.500016

"""

from alembic import op
import sqlalchemy as sa


revision = "97c225be0651"
down_revision = "5f8a87e796bb"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "transport_requests",
        sa.Column("external_carrier_email", sa.String(length=255), nullable=True),
    )


def downgrade():
    op.drop_column("transport_requests", "external_carrier_email")
