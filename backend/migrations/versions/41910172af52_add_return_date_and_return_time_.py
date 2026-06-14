
"""add_return_date_and_return_time_confirmed_to_transport_request

Revision ID: 41910172af52
Revises: b7ae9619aa32
Create Date: 2026-06-12 01:15:22.882998

"""
from alembic import op
import sqlalchemy as sa


revision = "41910172af52"
down_revision = "b7ae9619aa32"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "transport_requests",
        sa.Column("return_date", sa.Date(), nullable=True),
    )
    op.add_column(
        "transport_requests",
        sa.Column(
            "return_time_confirmed",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
    )


def downgrade():
    op.drop_column("transport_requests", "return_time_confirmed")
    op.drop_column("transport_requests", "return_date")
