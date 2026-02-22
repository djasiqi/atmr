"""Add location type and entry point columns to transport_requests.

New columns:
- pickup_type (VARCHAR(20), nullable) — institution | domicile | other
- dropoff_type (VARCHAR(20), nullable) — institution | domicile | other
- pickup_entry_point (VARCHAR(100), nullable) — Point d'accueil départ
- dropoff_entry_point (VARCHAR(100), nullable) — Point d'accueil arrivée

All columns are nullable for backward compatibility with existing requests.

Revision ID: transport_loc_001
Revises: transport_ux_001
Create Date: 2026-02-09
"""

from alembic import op

revision = "transport_loc_001"
down_revision = "transport_ux_001"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE transport_requests
        ADD COLUMN IF NOT EXISTS pickup_type VARCHAR(20)
    """)
    op.execute("""
        ALTER TABLE transport_requests
        ADD COLUMN IF NOT EXISTS dropoff_type VARCHAR(20)
    """)
    op.execute("""
        ALTER TABLE transport_requests
        ADD COLUMN IF NOT EXISTS pickup_entry_point VARCHAR(100)
    """)
    op.execute("""
        ALTER TABLE transport_requests
        ADD COLUMN IF NOT EXISTS dropoff_entry_point VARCHAR(100)
    """)

    # Add CHECK constraint for valid location types
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'chk_pickup_type_valid'
            ) THEN
                ALTER TABLE transport_requests
                ADD CONSTRAINT chk_pickup_type_valid
                CHECK (pickup_type IS NULL OR pickup_type IN ('institution', 'domicile', 'other'));
            END IF;
        END$$;
    """)
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'chk_dropoff_type_valid'
            ) THEN
                ALTER TABLE transport_requests
                ADD CONSTRAINT chk_dropoff_type_valid
                CHECK (dropoff_type IS NULL OR dropoff_type IN ('institution', 'domicile', 'other'));
            END IF;
        END$$;
    """)


def downgrade():
    op.execute(
        "ALTER TABLE transport_requests DROP CONSTRAINT IF EXISTS chk_dropoff_type_valid"
    )
    op.execute(
        "ALTER TABLE transport_requests DROP CONSTRAINT IF EXISTS chk_pickup_type_valid"
    )
    op.execute(
        "ALTER TABLE transport_requests DROP COLUMN IF EXISTS dropoff_entry_point"
    )
    op.execute(
        "ALTER TABLE transport_requests DROP COLUMN IF EXISTS pickup_entry_point"
    )
    op.execute("ALTER TABLE transport_requests DROP COLUMN IF EXISTS dropoff_type")
    op.execute("ALTER TABLE transport_requests DROP COLUMN IF EXISTS pickup_type")
