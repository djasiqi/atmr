
"""Merge all heads into single branch

Revision ID: 058be909ad9c
Revises: ('bp_ext_ref_idx_001', 'pat_logistics_001', 'transport_loc_001', 'invite_fields_001', '20260209_catchup')
Create Date: 2026-02-13 16:09:31.680076

"""
from alembic import op
import sqlalchemy as sa


revision = "058be909ad9c"
down_revision = ('bp_ext_ref_idx_001', 'pat_logistics_001', 'transport_loc_001', 'invite_fields_001', '20260209_catchup')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass

