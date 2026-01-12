"""Merge multiple migration branches

Revision ID: 94d80b1d14f0
Revises: ('add_credit_tip_partner_inv', 'fix_partnership_status_uppercase')
Create Date: 2026-01-09 12:05:46.924198

"""

from alembic import op
import sqlalchemy as sa


revision = "94d80b1d14f0"
down_revision = ("add_credit_tip_partner_inv", "fix_partnership_status_uppercase")
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
