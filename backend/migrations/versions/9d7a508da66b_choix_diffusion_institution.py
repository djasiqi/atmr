"""choix diffusion institution

Revision ID: 9d7a508da66b
Revises: c3a91e70b4f2
Create Date: 2026-10-21 07:35:59.331029

Colonne nullable ``dispatch_target`` sur les demandes institution.
La génération automatique a aussi vu d'autres écarts de modèle déjà couverts
par des révisions non appliquées sur cette base : ils sont exclus.
"""

from alembic import op
import sqlalchemy as sa


revision = "9d7a508da66b"
down_revision = "c3a91e70b4f2"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "transport_requests",
        sa.Column("dispatch_target", sa.String(length=20), nullable=True),
    )


def downgrade():
    op.drop_column("transport_requests", "dispatch_target")
