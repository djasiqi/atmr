
"""Merge migrations heads

Revision ID: 8186648ac54e
Revises: ('3_4_profiling', 'rl_suggestions_001', 'fix_audit_metadata')
Create Date: 2025-10-31 11:05:20.836158

"""
from alembic import op


revision = "8186648ac54e"
down_revision = ("3_4_profiling", "rl_suggestions_001", "fix_audit_metadata")
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass

