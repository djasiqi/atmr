"""attestation prestation trajet

Revision ID: f44ce1cd5095
Revises: f4c1a8e02b17
Create Date: 2026-10-25 00:26:45.608420

Migration limitée aux deux tables d'attestation. L'autogénération sur
l'arbre courant proposerait aussi des tables d'autres branches
(carnet privé, dossier de remboursement, etc.) absentes de la révision
f4c1a8e02b17 déjà appliquée sur atmr_test. Ces opérations ont été écartées.
"""

from alembic import op
import sqlalchemy as sa


revision = "f44ce1cd5095"
down_revision = "f4c1a8e02b17"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "transport_attestation",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("owner_company_id", sa.Integer(), nullable=False),
        sa.Column("executing_company_id", sa.Integer(), nullable=True),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("passenger_name", sa.String(length=120), nullable=False),
        sa.Column("pickup_location", sa.String(length=500), nullable=False),
        sa.Column("dropoff_location", sa.String(length=500), nullable=False),
        sa.Column("scheduled_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("arrived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("boarded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("outcome", sa.String(length=32), nullable=False),
        sa.Column("reason_code", sa.String(length=32), nullable=True),
        sa.Column("signature_sha256", sa.String(length=64), nullable=True),
        sa.Column("signature_b64", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column(
            "confirmed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
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
            "(outcome = 'signature_obtained' AND signature_sha256 IS NOT NULL) "
            "OR (outcome <> 'signature_obtained' AND signature_sha256 IS NULL)",
            name="ck_transport_attestation_signature",
        ),
        sa.CheckConstraint(
            "outcome IN ('signature_obtained', 'signature_refused', "
            "'signature_impossible', 'pending_verification')",
            name="ck_transport_attestation_outcome",
        ),
        sa.CheckConstraint(
            "reason_code IS NULL OR reason_code IN ('refused', 'unable', 'technical')",
            name="ck_transport_attestation_reason",
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["executing_company_id"], ["company.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["owner_company_id"], ["company.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
    )
    op.create_index(
        "ix_transport_attestation_booking_id",
        "transport_attestation",
        ["booking_id"],
        unique=True,
    )
    op.create_index(
        "ix_transport_attestation_company",
        "transport_attestation",
        ["owner_company_id"],
        unique=False,
    )
    op.create_index(
        "ix_transport_attestation_driver",
        "transport_attestation",
        ["driver_id"],
        unique=False,
    )
    op.create_table(
        "transport_attestation_event",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("attestation_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("outcome", sa.String(length=32), nullable=True),
        sa.Column("driver_id", sa.Integer(), nullable=True),
        sa.Column("detail", sa.Text(), server_default=sa.text("'{}'"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["attestation_id"], ["transport_attestation.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_transport_attestation_event_parent",
        "transport_attestation_event",
        ["attestation_id", "created_at"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_transport_attestation_event_parent",
        table_name="transport_attestation_event",
    )
    op.drop_table("transport_attestation_event")
    op.drop_index("ix_transport_attestation_driver", table_name="transport_attestation")
    op.drop_index(
        "ix_transport_attestation_company", table_name="transport_attestation"
    )
    op.drop_index(
        "ix_transport_attestation_booking_id", table_name="transport_attestation"
    )
    op.drop_table("transport_attestation")
