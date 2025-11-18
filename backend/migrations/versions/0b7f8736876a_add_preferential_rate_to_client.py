"""add_preferential_rate_to_client

Revision ID: 0b7f8736876a
Revises: d9f3a8b2c4e5
Create Date: 2025-10-10 21:46:48.744206

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0b7f8736876a"
down_revision = "d9f3a8b2c4e5"
branch_labels = None
depends_on = None


def upgrade():
    # Ajouter la colonne preferential_rate à la table client
    op.add_column("client", sa.Column("preferential_rate", sa.Numeric(10, 2), nullable=True))

    # Ajouter un commentaire pour documenter les valeurs typiques
    op.execute(
        "COMMENT ON COLUMN client.preferential_rate IS 'Tarif préférentiel en CHF (ex: 45, 50, 55, 60, 70, 80, 110). NULL = tarif standard'"
    )


def downgrade():
    # Supprimer la colonne preferential_rate
    op.drop_column("client", "preferential_rate")
