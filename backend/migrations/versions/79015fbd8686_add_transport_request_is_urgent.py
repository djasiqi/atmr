
"""add_transport_request_is_urgent

Revision ID: 79015fbd8686
Revises: b878346bc2ce
Create Date: 2026-06-22 19:56:48.718487

"""
from alembic import op
import sqlalchemy as sa


revision = "79015fbd8686"
down_revision = "b878346bc2ce"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "transport_requests",
        sa.Column(
            "is_urgent",
            sa.Boolean(),
            server_default="false",
            nullable=False,
            comment="Mission urgente (départ immédiat autorisé si pas de départ confirmé)",
        ),
    )


def downgrade():
    op.drop_column("transport_requests", "is_urgent")
