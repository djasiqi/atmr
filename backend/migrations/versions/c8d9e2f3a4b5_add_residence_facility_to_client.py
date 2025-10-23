"""add residence_facility to client

Revision ID: c8d9e2f3a4b5
Revises: af5e460cd09e
Create Date: 2025-10-17 22:10:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c8d9e2f3a4b5'
down_revision = 'af5e460cd09e'
branch_labels = None
depends_on = None


def upgrade():
    # Ajouter la colonne residence_facility à la table client
    op.add_column('client', sa.Column('residence_facility', sa.String(length=200), nullable=True))


def downgrade():
    # Supprimer la colonne residence_facility de la table client
    op.drop_column('client', 'residence_facility')

