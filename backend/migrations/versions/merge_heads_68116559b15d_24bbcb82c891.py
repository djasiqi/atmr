"""Merge heads 68116559b15d and 24bbcb82c891

Fusionne les deux branches de migration pour résoudre le conflit de heads.

Revision ID: merge_68116559b15d_24bbcb82c891
Revises: ('68116559b15d', '24bbcb82c891')
Create Date: 2025-11-27 16:55:00.000000

"""

# revision identifiers, used by Alembic.
revision = "merge_68116559b15d_24bbcb82c891"
down_revision = ("68116559b15d", "24bbcb82c891")
branch_labels = None
depends_on = None


def upgrade():
    """
    Merge migration - aucune modification de schéma nécessaire.
    Cette migration unifie simplement les deux branches de migration.
    """
    pass


def downgrade():
    """
    Downgrade - aucune modification de schéma nécessaire.
    """
    pass
