"""ajoute le type d assistance habituel sur le client

Revision ID: b2ca98b6cb2d
Revises: b79143d0fdc3
Create Date: 2026-09-29 18:08:44.366208

"""

from alembic import op
import sqlalchemy as sa


revision = "b2ca98b6cb2d"
down_revision = "b79143d0fdc3"
branch_labels = None
depends_on = None


def upgrade():
    # Autogenerate Alembic, puis limité à la colonne demandée.
    # Le comparateur voyait aussi des écarts d'index hors de ce changement.
    op.add_column(
        "client",
        sa.Column("habitual_assistance_detail", sa.String(length=200), nullable=True),
    )


def downgrade():
    op.drop_column("client", "habitual_assistance_detail")
