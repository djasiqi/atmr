"""Add linked_institution_id FK to client table.

Allows company clients (is_institution=true) to be formally linked
to an official Institution record on the platform.

Revision ID: linked_inst_001
Revises: (auto)
Create Date: 2026-02-11
"""

from alembic import op

# revision identifiers
revision = "linked_inst_001"
down_revision = "20260211_guard"
branch_labels = None
depends_on = None


def upgrade():
    # Step 1: Add nullable FK column
    op.execute("""
        ALTER TABLE client
        ADD COLUMN IF NOT EXISTS linked_institution_id INTEGER
        REFERENCES institutions(id) ON DELETE SET NULL
    """)

    # Step 2: Index for fast lookups (company_id, linked_institution_id)
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_client_company_linked_institution
        ON client (company_id, linked_institution_id)
        WHERE linked_institution_id IS NOT NULL
    """)


def downgrade():
    op.execute("DROP INDEX IF EXISTS ix_client_company_linked_institution")
    op.execute("ALTER TABLE client DROP COLUMN IF EXISTS linked_institution_id")
