"""Add transport UX settings (default_pickup_mode, entry_points, default_contact_phone).

Supports the form refactoring: institution vs domicile pickup mode,
configurable entry points (suggestions), and default contact phone.

Revision ID: transport_ux_001
Revises: (auto)
Create Date: 2026-02-09
"""

from alembic import op

revision = "transport_ux_001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE institution_settings
        ADD COLUMN IF NOT EXISTS default_pickup_mode VARCHAR(20) NOT NULL DEFAULT 'institution'
    """)
    op.execute("""
        ALTER TABLE institution_settings
        ADD COLUMN IF NOT EXISTS entry_points JSONB NOT NULL DEFAULT '[]'
    """)
    op.execute("""
        ALTER TABLE institution_settings
        ADD COLUMN IF NOT EXISTS default_contact_phone VARCHAR(50)
    """)


def downgrade():
    op.execute(
        "ALTER TABLE institution_settings DROP COLUMN IF EXISTS default_contact_phone"
    )
    op.execute("ALTER TABLE institution_settings DROP COLUMN IF EXISTS entry_points")
    op.execute(
        "ALTER TABLE institution_settings DROP COLUMN IF EXISTS default_pickup_mode"
    )
