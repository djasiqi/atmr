"""merge_device_tokens_and_enum_fix

Revision ID: 81ffc072d63d
Revises: ('4a4c71c80d0c', 'c0a12b09003e')
Create Date: 2026-01-15 03:42:55.849117

"""

from alembic import op
import sqlalchemy as sa


revision = "81ffc072d63d"
down_revision = "c0a12b09003e"  # Après fix_enum_and_index_naming seulement
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
