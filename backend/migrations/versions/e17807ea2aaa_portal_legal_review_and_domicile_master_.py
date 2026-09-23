"""portal legal review and domicile master data

Revision ID: e17807ea2aaa
Revises: 9daf0fb3cef7
Create Date: 2026-09-24 00:19:54.000000

Colonnes / table générées par ``flask db revision --autogenerate``.
Écarts hors périmètre retirés. Pas de ``batch_alter_table``.
Aucun backfill domicile sur créances historiques.
"""

import sqlalchemy as sa
from alembic import op

revision = "e17807ea2aaa"
down_revision = "9daf0fb3cef7"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "company",
        sa.Column("legal_name", sa.String(length=200), nullable=True),
    )
    op.add_column(
        "portal_receivable",
        sa.Column("debtor_domicile_address_snapshot", sa.Text(), nullable=True),
    )
    op.add_column(
        "portal_receivable",
        sa.Column("debtor_domicile_semantics", sa.String(length=40), nullable=True),
    )

    op.create_table(
        "portal_collection_legal_review",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("receivable_id", sa.Integer(), nullable=False),
        sa.Column("transmission_id", sa.Integer(), nullable=False),
        sa.Column("creditor_company_id", sa.Integer(), nullable=False),
        sa.Column("dossier_hash", sa.String(length=64), nullable=False),
        sa.Column("review_status", sa.String(length=32), nullable=False),
        sa.Column("review_version", sa.String(length=64), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("reviewed_by_user_id", sa.Integer(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "review_status IN ('pending', 'approved', 'rejected')",
            name="ck_portal_legal_review_status",
        ),
        sa.ForeignKeyConstraint(
            ["creditor_company_id"], ["company.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["receivable_id"], ["portal_receivable.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["reviewed_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["transmission_id"],
            ["portal_receivable_collection_transmission.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "transmission_id",
            "dossier_hash",
            "review_version",
            name="uq_portal_legal_review_tx_hash_ver",
        ),
    )
    op.create_index(
        "ix_portal_legal_review_receivable",
        "portal_collection_legal_review",
        ["receivable_id"],
        unique=False,
    )
    op.create_index(
        "ix_portal_legal_review_transmission",
        "portal_collection_legal_review",
        ["transmission_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_portal_legal_review_transmission",
        table_name="portal_collection_legal_review",
    )
    op.drop_index(
        "ix_portal_legal_review_receivable",
        table_name="portal_collection_legal_review",
    )
    op.drop_table("portal_collection_legal_review")
    op.drop_column("portal_receivable", "debtor_domicile_semantics")
    op.drop_column("portal_receivable", "debtor_domicile_address_snapshot")
    op.drop_column("company", "legal_name")
