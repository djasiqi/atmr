"""Add dedupe_key column to institution_notifications and company_notifications.

Fixes missing column that was in ORM models but not in original migrations.

Revision ID: 20260218_dedupe
Revises: 20260217_comp_notif
Create Date: 2026-02-18
"""

from alembic import op

revision = "20260218_dedupe"
down_revision = "20260218_drv_vehicle"
branch_labels = None
depends_on = None


def _add_dedupe(table: str, constraint: str, fk_col: str) -> None:
    """Add dedupe_key + unique constraint if not present."""
    op.execute(f"""
        ALTER TABLE {table}
        ADD COLUMN IF NOT EXISTS dedupe_key VARCHAR(200);
    """)
    op.execute(f"""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = '{constraint}'
            ) THEN
                ALTER TABLE {table}
                ADD CONSTRAINT {constraint}
                UNIQUE ({fk_col}, dedupe_key);
            END IF;
        END
        $$;
    """)


def upgrade():
    _add_dedupe(
        "institution_notifications", "uq_inst_notif_dedupe", "institution_id"
    )
    _add_dedupe(
        "company_notifications", "uq_comp_notif_dedupe", "company_id"
    )


def downgrade():
    for table, constraint in [
        ("institution_notifications", "uq_inst_notif_dedupe"),
        ("company_notifications", "uq_comp_notif_dedupe"),
    ]:
        op.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {constraint};")
        op.execute(f"ALTER TABLE {table} DROP COLUMN IF EXISTS dedupe_key;")
