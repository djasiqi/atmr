"""platform_issued_invoice_editor_lines_snapshot_1n

Revision ID: b5b935af8e86
Revises: e1ae4a70c23a
Create Date: 2026-08-04 15:20:00.505588

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "b5b935af8e86"
down_revision = "e1ae4a70c23a"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "platform_issued_invoice",
        sa.Column(
            "lines_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )
    op.add_column(
        "platform_issued_invoice",
        sa.Column("replace_idempotency_key", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "platform_issued_invoice",
        sa.Column("commercial_reference", sa.String(length=128), nullable=True),
    )

    op.drop_index(
        "uq_platform_issued_invoice_statement",
        table_name="platform_issued_invoice",
    )
    op.create_index(
        "ix_platform_issued_invoice_statement_id",
        "platform_issued_invoice",
        ["statement_id"],
        unique=False,
    )
    op.create_index(
        "uq_platform_issued_active_statement",
        "platform_issued_invoice",
        ["statement_id"],
        unique=True,
        postgresql_where=sa.text(
            "statement_id IS NOT NULL "
            "AND document_type = 'INVOICE' "
            "AND status NOT IN ('CANCELLED', 'CREDITED')"
        ),
    )
    op.create_index(
        "uq_platform_issued_replaces",
        "platform_issued_invoice",
        ["replaces_issued_invoice_id"],
        unique=True,
        postgresql_where=sa.text("replaces_issued_invoice_id IS NOT NULL"),
    )
    op.create_index(
        "uq_platform_issued_replace_idempotency",
        "platform_issued_invoice",
        ["replace_idempotency_key"],
        unique=True,
        postgresql_where=sa.text("replace_idempotency_key IS NOT NULL"),
    )

    # Backfill lines_snapshot depuis les lignes du relevé lié
    op.execute(
        """
        UPDATE platform_issued_invoice AS inv
        SET lines_snapshot = sub.lines
        FROM (
            SELECT
                inv2.id AS issued_id,
                COALESCE(
                    (
                        SELECT jsonb_agg(
                            jsonb_build_object(
                                'line_type', ln.line_type,
                                'label', ln.label,
                                'quantity', ln.quantity,
                                'unit_amount', ln.unit_amount,
                                'amount', ln.amount,
                                'calculation_mode',
                                    CASE
                                        WHEN ln.quantity IS NOT NULL
                                             AND ln.unit_amount IS NOT NULL
                                        THEN 'UNIT_PRICE'
                                        ELSE 'FIXED_AMOUNT'
                                    END
                            )
                            ORDER BY ln.sort_order, ln.id
                        )
                        FROM platform_invoice_line ln
                        WHERE ln.invoice_id = inv2.statement_id
                    ),
                    '[]'::jsonb
                ) AS lines
            FROM platform_issued_invoice inv2
            WHERE inv2.statement_id IS NOT NULL
              AND inv2.lines_snapshot IS NULL
        ) AS sub
        WHERE inv.id = sub.issued_id
        """
    )


def downgrade():
    op.drop_index(
        "uq_platform_issued_replace_idempotency",
        table_name="platform_issued_invoice",
        postgresql_where=sa.text("replace_idempotency_key IS NOT NULL"),
    )
    op.drop_index(
        "uq_platform_issued_replaces",
        table_name="platform_issued_invoice",
        postgresql_where=sa.text("replaces_issued_invoice_id IS NOT NULL"),
    )
    op.drop_index(
        "uq_platform_issued_active_statement",
        table_name="platform_issued_invoice",
        postgresql_where=sa.text(
            "statement_id IS NOT NULL "
            "AND document_type = 'INVOICE' "
            "AND status NOT IN ('CANCELLED', 'CREDITED')"
        ),
    )
    op.drop_index(
        "ix_platform_issued_invoice_statement_id",
        table_name="platform_issued_invoice",
    )
    op.create_index(
        "uq_platform_issued_invoice_statement",
        "platform_issued_invoice",
        ["statement_id"],
        unique=True,
    )
    op.drop_column("platform_issued_invoice", "commercial_reference")
    op.drop_column("platform_issued_invoice", "replace_idempotency_key")
    op.drop_column("platform_issued_invoice", "lines_snapshot")
