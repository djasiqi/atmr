"""Fix dispatch mode enum values to uppercase

Revision ID: af5e460cd09e
Revises: a8f4c9e2b1d3
Create Date: 2025-10-17 15:24:09.359623

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "af5e460cd09e"
down_revision = "a8f4c9e2b1d3"
branch_labels = None
depends_on = None


def upgrade():
    """
    Corrige les valeurs de l'enum dispatchmode pour utiliser UPPERCASE.
    PostgreSQL a créé l'enum avec MANUAL, SEMI_AUTO, FULLY_AUTO mais
    les valeurs par défaut étaient en lowercase.
    """
    # Mettre à jour les valeurs existantes en UPPERCASE
    op.execute("UPDATE company SET dispatch_mode = 'SEMI_AUTO' WHERE dispatch_mode::text = 'semi_auto'")
    op.execute("UPDATE company SET dispatch_mode = 'MANUAL' WHERE dispatch_mode::text = 'manual'")
    op.execute("UPDATE company SET dispatch_mode = 'FULLY_AUTO' WHERE dispatch_mode::text = 'fully_auto'")


def downgrade():
    """
    Revenir aux valeurs lowercase (si nécessaire).
    """
    op.execute("UPDATE company SET dispatch_mode = 'semi_auto' WHERE dispatch_mode::text = 'SEMI_AUTO'")
    op.execute("UPDATE company SET dispatch_mode = 'manual' WHERE dispatch_mode::text = 'MANUAL'")
    op.execute("UPDATE company SET dispatch_mode = 'fully_auto' WHERE dispatch_mode::text = 'FULLY_AUTO'")
