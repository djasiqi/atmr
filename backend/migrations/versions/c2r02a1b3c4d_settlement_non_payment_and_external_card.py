"""settlement non-payment incidents + external card terminal

Revision ID: c2r02a1b3c4d
Revises: 38c76fa54225
Create Date: 2026-11-14 13:30:00.000000

Migration limitée aux tables CONVENTION-REGLEMENT-02.
L'autogénération sur l'arbre courant proposerait aussi des écarts
hors périmètre ; ces opérations ont été écartées.
"""

from alembic import op
import sqlalchemy as sa


revision = "c2r02a1b3c4d"
down_revision = "38c76fa54225"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "settlement_non_payment_incident",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("client_id", sa.Integer(), nullable=True),
        sa.Column("debtor_id", sa.Integer(), nullable=True),
        sa.Column("debtor_label", sa.String(length=160), nullable=True),
        sa.Column("amount_due_chf", sa.Numeric(10, 2), nullable=False),
        sa.Column(
            "currency",
            sa.String(length=3),
            server_default=sa.text("'CHF'"),
            nullable=False,
        ),
        sa.Column("transport_date", sa.String(length=32), nullable=False),
        sa.Column("motif", sa.String(length=64), nullable=False),
        sa.Column(
            "observation", sa.Text(), server_default=sa.text("''"), nullable=False
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default=sa.text("'a_traiter'"),
            nullable=False,
        ),
        sa.Column(
            "payment_recorded",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "signature_recorded",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "deferred_billing_authorized",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "mission_completed",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "mission_close_allowed",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ("
            "'a_traiter', 'en_cours_examen', 'regularise', 'cloture_justifie')",
            name="ck_snp_incident_status",
        ),
        sa.CheckConstraint(
            "motif IN ("
            "'refus_signer_et_payer', 'impossibilite_regler', "
            "'contestation_montant', 'moyen_paiement_indisponible_ou_echoue', "
            "'autre')",
            name="ck_snp_incident_motif",
        ),
        sa.CheckConstraint("currency = 'CHF'", name="ck_snp_incident_currency"),
        sa.CheckConstraint("amount_due_chf >= 0", name="ck_snp_incident_amount"),
        sa.CheckConstraint(
            "payment_recorded = false", name="ck_snp_incident_not_payment"
        ),
        sa.CheckConstraint(
            "signature_recorded = false", name="ck_snp_incident_not_signature"
        ),
        sa.CheckConstraint(
            "deferred_billing_authorized = false",
            name="ck_snp_incident_not_deferred",
        ),
        sa.CheckConstraint(
            "mission_completed = true", name="ck_snp_incident_mission_completed"
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["client_id"], ["client.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id", name="uq_snp_incident_public_id"),
        sa.UniqueConstraint(
            "company_id", "idempotency_key", name="uq_snp_incident_company_key"
        ),
    )
    op.create_index(
        "ix_snp_incident_company_status",
        "settlement_non_payment_incident",
        ["company_id", "status"],
    )
    op.create_index(
        "ix_snp_incident_booking",
        "settlement_non_payment_incident",
        ["booking_id"],
    )

    op.create_table(
        "settlement_non_payment_event",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("incident_id", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column("actor_role", sa.String(length=32), nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("detail", sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("char_length(idempotency_key) > 0", name="ck_snp_event_key"),
        sa.ForeignKeyConstraint(
            ["incident_id"],
            ["settlement_non_payment_incident.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "incident_id", "idempotency_key", name="uq_snp_event_incident_key"
        ),
    )
    op.create_index(
        "ix_snp_event_incident",
        "settlement_non_payment_event",
        ["incident_id", "id"],
    )

    op.create_table(
        "external_card_settlement",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("amount_chf", sa.Numeric(10, 2), nullable=False),
        sa.Column(
            "currency",
            sa.String(length=3),
            server_default=sa.text("'CHF'"),
            nullable=False,
        ),
        sa.Column(
            "channel",
            sa.String(length=32),
            server_default=sa.text("'external_terminal'"),
            nullable=False,
        ),
        sa.Column(
            "tap_to_pay",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "provider_name",
            sa.String(length=64),
            server_default=sa.text("'external_bank_terminal'"),
            nullable=False,
        ),
        sa.Column(
            "reference",
            sa.String(length=120),
            server_default=sa.text("''"),
            nullable=False,
        ),
        sa.Column(
            "receipt_note", sa.Text(), server_default=sa.text("''"), nullable=False
        ),
        sa.Column(
            "confirmed_by_driver",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("currency = 'CHF'", name="ck_ext_card_currency"),
        sa.CheckConstraint("amount_chf > 0", name="ck_ext_card_amount"),
        sa.CheckConstraint("channel = 'external_terminal'", name="ck_ext_card_channel"),
        sa.CheckConstraint("tap_to_pay = false", name="ck_ext_card_not_tap"),
        sa.CheckConstraint(
            "provider_name = 'external_bank_terminal'", name="ck_ext_card_provider"
        ),
        sa.CheckConstraint(
            "char_length(reference) > 0 OR char_length(receipt_note) > 0",
            name="ck_ext_card_proof",
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "company_id", "idempotency_key", name="uq_ext_card_company_key"
        ),
    )
    op.create_index("ix_ext_card_booking", "external_card_settlement", ["booking_id"])


def downgrade():
    op.drop_index("ix_ext_card_booking", table_name="external_card_settlement")
    op.drop_table("external_card_settlement")
    op.drop_index("ix_snp_event_incident", table_name="settlement_non_payment_event")
    op.drop_table("settlement_non_payment_event")
    op.drop_index(
        "ix_snp_incident_booking", table_name="settlement_non_payment_incident"
    )
    op.drop_index(
        "ix_snp_incident_company_status", table_name="settlement_non_payment_incident"
    )
    op.drop_table("settlement_non_payment_incident")
