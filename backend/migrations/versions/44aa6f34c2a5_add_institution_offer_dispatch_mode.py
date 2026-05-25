
"""add institution offer dispatch mode

Revision ID: 44aa6f34c2a5
Revises: 20260525_req_ext_ref_optional
Create Date: 2026-05-25 17:17:36.002619

"""
from alembic import op
import sqlalchemy as sa


revision = "44aa6f34c2a5"
down_revision = "20260525_req_ext_ref_optional"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "institution_settings",
        sa.Column(
            "offer_dispatch_mode",
            sa.String(length=20),
            nullable=False,
            server_default="sequential",
            comment="Mode d'envoi des demandes: sequential | broadcast",
        ),
    )


def downgrade():
    op.drop_column("institution_settings", "offer_dispatch_mode")

