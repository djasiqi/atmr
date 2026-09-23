"""portal collection transmission evidence 6G-B

Revision ID: 773df2373b29
Revises: e17807ea2aaa
Create Date: 2026-09-24 00:31:47.189441

Tables / colonnes générées par ``flask db revision --autogenerate``.
Écarts d'index hors périmètre retirés. Pas de ``batch_alter_table``.
Aucun backfill : les drafts historiques restent non transmis.
"""

import sqlalchemy as sa
from alembic import op

revision = "773df2373b29"
down_revision = "e17807ea2aaa"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "portal_collection_transmission_evidence",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("transmission_id", sa.Integer(), nullable=False),
        sa.Column("receivable_id", sa.Integer(), nullable=False),
        sa.Column("creditor_company_id", sa.Integer(), nullable=False),
        sa.Column("dossier_hash", sa.String(length=64), nullable=False),
        sa.Column("transmission_kind", sa.String(length=40), nullable=False),
        sa.Column("channel", sa.String(length=64), nullable=False),
        sa.Column("event_kind", sa.String(length=32), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("recorded_by_user_id", sa.Integer(), nullable=False),
        sa.Column("recipient", sa.String(length=255), nullable=True),
        sa.Column("external_reference", sa.String(length=200), nullable=True),
        sa.Column("evidence_type", sa.String(length=64), nullable=False),
        sa.Column("evidence_payload", sa.Text(), nullable=False),
        sa.Column("evidence_hash", sa.String(length=64), nullable=False),
        sa.Column("acknowledgment_reference", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "channel IN ("
            "'manual_office_submission', 'registered_mail', "
            "'email', 'collection_provider_manual', 'local_artifact'"
            ")",
            name="ck_portal_tx_evidence_channel",
        ),
        sa.CheckConstraint(
            "event_kind IN ('export_prepared', 'transmitted', 'acknowledged')",
            name="ck_portal_tx_evidence_event",
        ),
        sa.CheckConstraint(
            "transmission_kind IN ('PURSUIT', 'PRIVATE_COLLECTION')",
            name="ck_portal_tx_evidence_kind",
        ),
        sa.ForeignKeyConstraint(
            ["creditor_company_id"], ["company.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["receivable_id"], ["portal_receivable.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["recorded_by_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["transmission_id"],
            ["portal_receivable_collection_transmission.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_portal_tx_evidence_hash",
        "portal_collection_transmission_evidence",
        ["dossier_hash"],
        unique=False,
    )
    op.create_index(
        "ix_portal_tx_evidence_receivable",
        "portal_collection_transmission_evidence",
        ["receivable_id"],
        unique=False,
    )
    op.create_index(
        "ix_portal_tx_evidence_transmission",
        "portal_collection_transmission_evidence",
        ["transmission_id"],
        unique=False,
    )

    op.add_column(
        "portal_receivable_collection_transmission",
        sa.Column("recipient_label", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "portal_receivable_collection_transmission",
        sa.Column("recipient_contact", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "portal_receivable_collection_transmission",
        sa.Column(
            "recipient_summary_confirmed",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )
    op.add_column(
        "portal_receivable_collection_transmission",
        sa.Column(
            "recipient_confirmed_at", sa.DateTime(timezone=True), nullable=True
        ),
    )
    op.add_column(
        "portal_receivable_collection_transmission",
        sa.Column("recipient_confirmed_by_user_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "portal_receivable_collection_transmission",
        sa.Column("export_version", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "portal_receivable_collection_transmission",
        sa.Column(
            "export_prepared_at", sa.DateTime(timezone=True), nullable=True
        ),
    )
    op.create_foreign_key(
        "fk_portal_coll_tx_recipient_confirmed_by",
        "portal_receivable_collection_transmission",
        "user",
        ["recipient_confirmed_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )

    op.drop_constraint(
        "ck_portal_coll_action_type",
        "portal_receivable_collection_action",
        type_="check",
    )
    op.create_check_constraint(
        "ck_portal_coll_action_type",
        "portal_receivable_collection_action",
        "action_type IN ("
        "'PURSUIT_DRAFT_PREPARED', "
        "'PRIVATE_COLLECTION_DRAFT_PREPARED', "
        "'EXPORT_PREPARED', "
        "'TRANSMISSION_AUTHORIZED', "
        "'TRANSMITTED', "
        "'ACKNOWLEDGED', "
        "'TRANSMISSION_CANCELLED'"
        ")",
    )


def downgrade():
    op.drop_constraint(
        "ck_portal_coll_action_type",
        "portal_receivable_collection_action",
        type_="check",
    )
    op.create_check_constraint(
        "ck_portal_coll_action_type",
        "portal_receivable_collection_action",
        "action_type IN ("
        "'PURSUIT_DRAFT_PREPARED', "
        "'PRIVATE_COLLECTION_DRAFT_PREPARED', "
        "'TRANSMISSION_CANCELLED'"
        ")",
    )

    op.drop_constraint(
        "fk_portal_coll_tx_recipient_confirmed_by",
        "portal_receivable_collection_transmission",
        type_="foreignkey",
    )
    op.drop_column(
        "portal_receivable_collection_transmission", "export_prepared_at"
    )
    op.drop_column(
        "portal_receivable_collection_transmission", "export_version"
    )
    op.drop_column(
        "portal_receivable_collection_transmission",
        "recipient_confirmed_by_user_id",
    )
    op.drop_column(
        "portal_receivable_collection_transmission", "recipient_confirmed_at"
    )
    op.drop_column(
        "portal_receivable_collection_transmission",
        "recipient_summary_confirmed",
    )
    op.drop_column(
        "portal_receivable_collection_transmission", "recipient_contact"
    )
    op.drop_column(
        "portal_receivable_collection_transmission", "recipient_label"
    )

    op.drop_index(
        "ix_portal_tx_evidence_transmission",
        table_name="portal_collection_transmission_evidence",
    )
    op.drop_index(
        "ix_portal_tx_evidence_receivable",
        table_name="portal_collection_transmission_evidence",
    )
    op.drop_index(
        "ix_portal_tx_evidence_hash",
        table_name="portal_collection_transmission_evidence",
    )
    op.drop_table("portal_collection_transmission_evidence")
