"""Merge device_token_provider and refresh_token_soft_rotation heads

Revision ID: 20260222_merge
Revises: ('20260221_dt_prov', '20260221_soft_rot')
Create Date: 2026-02-22 23:50:00.000000

"""

revision = "20260222_merge"
down_revision = ("20260221_dt_prov", "20260221_soft_rot")
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
