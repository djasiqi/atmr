"""add_gps_coords_to_client

Revision ID: d8a4d7185c60
Revises: 715e89e538c3
Create Date: 2025-10-14 09:46:56.571233

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "d8a4d7185c60"
down_revision = "715e89e538c3"
branch_labels = None
depends_on = None


def upgrade():
    # Ajouter les colonnes GPS pour l'adresse de domicile
    op.add_column("client", sa.Column("domicile_lat", sa.Numeric(precision=10, scale=7), nullable=True))
    op.add_column("client", sa.Column("domicile_lon", sa.Numeric(precision=10, scale=7), nullable=True))

    # Ajouter les colonnes GPS pour l'adresse de facturation
    op.add_column("client", sa.Column("billing_lat", sa.Numeric(precision=10, scale=7), nullable=True))
    op.add_column("client", sa.Column("billing_lon", sa.Numeric(precision=10, scale=7), nullable=True))


def downgrade():
    # Supprimer les colonnes GPS
    op.drop_column("client", "billing_lon")
    op.drop_column("client", "billing_lat")
    op.drop_column("client", "domicile_lon")
    op.drop_column("client", "domicile_lat")
