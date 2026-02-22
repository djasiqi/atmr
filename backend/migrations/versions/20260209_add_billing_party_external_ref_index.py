"""Add unique partial index on billing_parties(company_id, external_ref).

Ensures no duplicate external_ref per company (e.g. "institution:42", "patient:123").
Partial index: only applies where external_ref IS NOT NULL.

Includes a dedup step to handle any pre-existing duplicates safely.

Revision ID: bp_ext_ref_idx_001
Revises: (auto)
Create Date: 2026-02-09
"""

from alembic import op

# revision identifiers
revision = "bp_ext_ref_idx_001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Step 1: Deduplicate external_ref within each company.
    # Keep the most recently updated row, deactivate older duplicates
    # by setting their external_ref to NULL (preserving the row).
    op.execute("""
        UPDATE billing_parties
        SET external_ref = NULL
        WHERE id IN (
            SELECT bp.id
            FROM billing_parties bp
            INNER JOIN (
                SELECT company_id, external_ref, MAX(id) AS keep_id
                FROM billing_parties
                WHERE external_ref IS NOT NULL
                GROUP BY company_id, external_ref
                HAVING COUNT(*) > 1
            ) dups
            ON bp.company_id = dups.company_id
            AND bp.external_ref = dups.external_ref
            AND bp.id != dups.keep_id
        )
    """)

    # Step 2: Create unique partial index (idempotent)
    op.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_billing_parties_company_external_ref
        ON billing_parties (company_id, external_ref)
        WHERE external_ref IS NOT NULL
    """)


def downgrade():
    op.execute("""
        DROP INDEX IF EXISTS uq_billing_parties_company_external_ref
    """)
