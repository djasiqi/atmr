"""Ajoute company.platform_suspended (gouvernance plateforme V1).

Revision ID: 20260328_co_plat_susp
Revises: 20260328_dt_push_lifecycle
Create Date: 2026-03-28

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "20260328_co_plat_susp"
down_revision = "20260328_dt_push_lifecycle"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "company",
        sa.Column(
            "platform_suspended",
            sa.Boolean(),
            nullable=False,
            server_default="false",
            comment="Intention persistée : tenant suspendu au sens plateforme",
        ),
    )


def downgrade() -> None:
    op.drop_column("company", "platform_suspended")
