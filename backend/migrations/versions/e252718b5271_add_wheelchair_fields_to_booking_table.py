"""Add wheelchair fields to booking table

Revision ID: e252718b5271
Revises: 0b7f8736876a
Create Date: 2025-10-11 00:07:12.184913

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e252718b5271"
down_revision = "0b7f8736876a"
branch_labels = None
depends_on = None


def upgrade():
    # Ajouter les colonnes wheelchair_client_has et wheelchair_need à la table booking
    op.add_column(
        "booking", sa.Column("wheelchair_client_has", sa.Boolean(), nullable=False, server_default=sa.text("false"))
    )
    op.add_column(
        "booking", sa.Column("wheelchair_need", sa.Boolean(), nullable=False, server_default=sa.text("false"))
    )


def downgrade():
    # Supprimer les colonnes wheelchair_client_has et wheelchair_need de la table booking
    op.drop_column("booking", "wheelchair_need")
    op.drop_column("booking", "wheelchair_client_has")
