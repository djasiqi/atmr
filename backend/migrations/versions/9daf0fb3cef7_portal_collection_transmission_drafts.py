"""portal collection transmission drafts

Revision ID: 9daf0fb3cef7
Revises: f1b95189e7f5
Create Date: 2026-09-24 00:07:14.934915

Tables générées par ``flask db revision --autogenerate``.
Écarts d'index hors périmètre retirés. Pas de ``batch_alter_table``.
"""

import sqlalchemy as sa
from alembic import op

revision = "9daf0fb3cef7"
down_revision = "f1b95189e7f5"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "portal_receivable_collection_transmission",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("receivable_id", sa.Integer(), nullable=False),
        sa.Column("creditor_company_id", sa.Integer(), nullable=False),
        sa.Column("transmission_type", sa.String(length=40), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("collection_prepared_event_id", sa.Integer(), nullable=True),
        sa.Column("creditor_snapshot", sa.Text(), nullable=False),
        sa.Column("debtor_snapshot", sa.Text(), nullable=False),
        sa.Column(
            "claim_principal_snapshot",
            sa.Numeric(precision=12, scale=2),
            nullable=False,
        ),
        sa.Column("payments_snapshot", sa.Text(), nullable=False),
        sa.Column(
            "balance_snapshot", sa.Numeric(precision=12, scale=2), nullable=False
        ),
        sa.Column("currency_snapshot", sa.String(length=3), nullable=False),
        sa.Column(
            "invoice_reference_snapshot", sa.String(length=80), nullable=False
        ),
        sa.Column("claim_reason_snapshot", sa.Text(), nullable=False),
        sa.Column("due_date_snapshot", sa.DateTime(timezone=True), nullable=False),
        sa.Column("export_payload", sa.Text(), nullable=False),
        sa.Column("export_hash", sa.String(length=64), nullable=False),
        sa.Column("pursuit_jurisdiction", sa.String(length=120), nullable=True),
        sa.Column("creditor_confirmed", sa.Boolean(), nullable=False),
        sa.Column(
            "requested_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("requested_by_user_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "currency_snapshot = 'CHF'", name="ck_portal_coll_tx_chf"
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'cancelled')", name="ck_portal_coll_tx_status"
        ),
        sa.CheckConstraint(
            "transmission_type IN ('PRIVATE_COLLECTION', 'PURSUIT_DRAFT')",
            name="ck_portal_coll_tx_type",
        ),
        sa.ForeignKeyConstraint(
            ["collection_prepared_event_id"],
            ["portal_receivable_dunning_event.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["creditor_company_id"], ["company.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["receivable_id"], ["portal_receivable.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["requested_by_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_portal_coll_tx_creditor",
        "portal_receivable_collection_transmission",
        ["creditor_company_id"],
        unique=False,
    )
    op.create_index(
        "ix_portal_coll_tx_receivable",
        "portal_receivable_collection_transmission",
        ["receivable_id"],
        unique=False,
    )

    op.create_table(
        "portal_receivable_collection_action",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("receivable_id", sa.Integer(), nullable=False),
        sa.Column("creditor_company_id", sa.Integer(), nullable=False),
        sa.Column("transmission_id", sa.Integer(), nullable=True),
        sa.Column("action_type", sa.String(length=64), nullable=False),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("requested_by_user_id", sa.Integer(), nullable=False),
        sa.Column("payload_snapshot", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "action_type IN ("
            "'PURSUIT_DRAFT_PREPARED', "
            "'PRIVATE_COLLECTION_DRAFT_PREPARED', "
            "'TRANSMISSION_CANCELLED'"
            ")",
            name="ck_portal_coll_action_type",
        ),
        sa.ForeignKeyConstraint(
            ["creditor_company_id"], ["company.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["receivable_id"], ["portal_receivable.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["requested_by_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["transmission_id"],
            ["portal_receivable_collection_transmission.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_portal_coll_action_creditor",
        "portal_receivable_collection_action",
        ["creditor_company_id"],
        unique=False,
    )
    op.create_index(
        "ix_portal_coll_action_receivable",
        "portal_receivable_collection_action",
        ["receivable_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_portal_coll_action_receivable",
        table_name="portal_receivable_collection_action",
    )
    op.drop_index(
        "ix_portal_coll_action_creditor",
        table_name="portal_receivable_collection_action",
    )
    op.drop_table("portal_receivable_collection_action")
    op.drop_index(
        "ix_portal_coll_tx_receivable",
        table_name="portal_receivable_collection_transmission",
    )
    op.drop_index(
        "ix_portal_coll_tx_creditor",
        table_name="portal_receivable_collection_transmission",
    )
    op.drop_table("portal_receivable_collection_transmission")
