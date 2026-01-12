"""Add avs_number to client

Revision ID: bf7baf7a0f6f
Revises: 2d6acf22f1f6
Create Date: 2026-01-10 18:04:24.686367

"""

from alembic import op
import sqlalchemy as sa


revision = "bf7baf7a0f6f"
down_revision = "45e9e93acca8"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "client", sa.Column("avs_number", sa.String(length=20), nullable=True)
    )


def downgrade():
    op.drop_column("client", "avs_number")
