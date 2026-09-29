"""ajoute le type d assistance sur la reservation

Revision ID: b79143d0fdc3
Revises: d98f32ddd475
Create Date: 2026-09-29 17:01:24.446309

"""
from alembic import op
import sqlalchemy as sa


revision = "b79143d0fdc3"
down_revision = "d98f32ddd475"
branch_labels = None
depends_on = None


def upgrade():
    # Autogenerate Alembic, puis limité à la colonne demandée.
    # Le comparateur voyait aussi des écarts d'index hors de ce changement.
    op.add_column(
        "booking",
        sa.Column("assistance_detail", sa.String(length=200), nullable=True),
    )


def downgrade():
    op.drop_column("booking", "assistance_detail")
