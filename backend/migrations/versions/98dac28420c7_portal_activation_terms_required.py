"""portal activation terms required

Revision ID: 98dac28420c7
Revises: 233355ed2b10
Create Date: 2026-09-23 21:04:47.108737

"""

import sqlalchemy as sa
from alembic import op

revision = "98dac28420c7"
down_revision = "233355ed2b10"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "activation_session",
        sa.Column(
            "portal_terms_required",
            sa.Boolean(),
            server_default="false",
            nullable=False,
        ),
    )


def downgrade():
    op.drop_column("activation_session", "portal_terms_required")
