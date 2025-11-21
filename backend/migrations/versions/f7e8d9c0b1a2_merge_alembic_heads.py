"""Merge multiple heads: a1b2c3d4e5f6, add_message_file_fields, b2c3d4e5f6a7

Unifie les 3 heads de migration pour permettre un upgrade sans ambiguïté.

Revision ID: f7e8d9c0b1a2
Revises: ('a1b2c3d4e5f6', 'add_message_file_fields', 'b2c3d4e5f6a7')
Create Date: 2025-01-28 12:00:00.000000

"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "f7e8d9c0b1a2"
down_revision = ("a1b2c3d4e5f6", "add_message_file_fields", "b2c3d4e5f6a7")
branch_labels = None
depends_on = None


def upgrade():
    """
    Merge migration - aucune modification de schéma nécessaire.
    Cette migration unifie simplement les branches de migration.
    """
    pass


def downgrade():
    """
    Downgrade - aucune modification de schéma nécessaire.
    """
    pass

