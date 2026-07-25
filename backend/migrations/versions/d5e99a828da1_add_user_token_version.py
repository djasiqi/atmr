"""add_user_token_version

Revision ID: d5e99a828da1
Revises: bd7f3de53566
Create Date: 2026-07-25 15:39:48.956444

"""

from alembic import op
import sqlalchemy as sa


revision = "d5e99a828da1"
down_revision = "bd7f3de53566"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "user",
        sa.Column(
            "token_version",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    )


def downgrade():
    op.drop_column("user", "token_version")
