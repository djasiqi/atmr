"""Add resource_type, resource_id, correlation_id to audit_logs + composite index.

Revision ID: 20260220_audit_res
Revises: 20260219_veh_ins_name
Create Date: 2026-02-20
"""

from alembic import op

revision = "20260220_audit_res"
down_revision = "20260219_veh_ins_name"
branch_labels = None
depends_on = None


def upgrade():
    op.execute(
        "ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS resource_type VARCHAR(50)"
    )
    op.execute(
        "ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS resource_id VARCHAR(64)"
    )
    op.execute(
        "ALTER TABLE audit_logs ADD COLUMN IF NOT EXISTS correlation_id VARCHAR(100)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_audit_logs_company_category_created "
        "ON audit_logs (company_id, action_category, created_at)"
    )


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_audit_logs_company_category_created")
    op.execute("ALTER TABLE audit_logs DROP COLUMN IF EXISTS correlation_id")
    op.execute("ALTER TABLE audit_logs DROP COLUMN IF EXISTS resource_id")
    op.execute("ALTER TABLE audit_logs DROP COLUMN IF EXISTS resource_type")
