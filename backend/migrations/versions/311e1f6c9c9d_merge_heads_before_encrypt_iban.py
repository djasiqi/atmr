"""merge_heads_before_encrypt_iban

Revision ID: 311e1f6c9c9d
Revises: None
Create Date: 2025-12-09 00:07:01.078867

"""

from alembic import op
import sqlalchemy as sa


revision = "311e1f6c9c9d"
down_revision = ("930de29a8cae", "p1_assignment_indexes")  # Merge des deux heads
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
