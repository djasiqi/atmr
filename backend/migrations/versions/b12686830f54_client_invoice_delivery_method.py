"""client_invoice_delivery_method

Revision ID: b12686830f54
Revises: 3041a7d1f098
Create Date: 2026-09-24 23:03:35.499854

"""

from alembic import op
import sqlalchemy as sa


revision = "b12686830f54"
down_revision = "3041a7d1f098"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "client",
        sa.Column(
            "invoice_delivery_method",
            sa.String(length=20),
            server_default="email",
            nullable=False,
        ),
    )


def downgrade():
    op.drop_column("client", "invoice_delivery_method")
