"""merge heads

Revision ID: d73f133dd06f
Revises: ('019e8e5179d9', 'merge_68116559b15d_24bbcb82c891')
Create Date: 2025-12-02 23:58:20.015971

"""

from alembic import op
import sqlalchemy as sa


revision = "d73f133dd06f"
down_revision = ("019e8e5179d9", "merge_68116559b15d_24bbcb82c891")
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
