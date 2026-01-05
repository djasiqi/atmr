"""Merge heads: a1b2c3d4e5f7 and p1a2r3t4n5s6

Unifie les 2 heads de migration pour permettre un upgrade sans ambiguïté.

Revision ID: m1e2r3g4e5h6
Revises: ('a1b2c3d4e5f7', 'p1a2r3t4n5s6')
Create Date: 2025-12-17 15:00:00.000000

"""

# revision identifiers, used by Alembic.
revision = "m1e2r3g4e5h6"
down_revision = ("a1b2c3d4e5f7", "p1a2r3t4n5s6")
branch_labels = None
depends_on = None


def upgrade():
    """
    Merge migration - aucune modification de schéma nécessaire.
    Cette migration unifie simplement les branches de migration.
    """


def downgrade():
    """
    Downgrade - aucune modification de schéma nécessaire.
    """
