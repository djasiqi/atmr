"""add postgis geom to geo_unit

Revision ID: 20260302_geo_unit_geom
Revises: 20260226_platform_zone_sets
Create Date: 2026-03-02 09:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260302_geo_unit_geom"
down_revision = "20260226_platform_zone_sets"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute(
        "ALTER TABLE geo_unit ADD COLUMN IF NOT EXISTS geom geometry(MultiPolygon, 4326)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_geo_unit_geom ON geo_unit USING GIST (geom)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_platform_zone_membership_zone_set_zone "
        "ON platform_zone_membership (zone_set_id, zone_id)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_platform_zone_membership_zone_set_zone")
    op.execute("DROP INDEX IF EXISTS ix_geo_unit_geom")
    op.execute("ALTER TABLE geo_unit DROP COLUMN IF EXISTS geom")
