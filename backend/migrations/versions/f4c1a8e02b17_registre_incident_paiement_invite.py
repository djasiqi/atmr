"""registre incident paiement invite

Revision ID: f4c1a8e02b17
Revises: e3a22ba34db5
Create Date: 2026-10-24 23:50:00.000000

Migration limitée aux deux tables du registre. L'autogénération sur
l'arbre courant proposerait aussi des tables d'autres branches
(dossier de remboursement, carnet privé, etc.) absentes de la révision
e3a22ba34db5 déjà appliquée sur atmr_test. Ces opérations ont été écartées.
"""

from alembic import op
import sqlalchemy as sa


revision = "f4c1a8e02b17"
down_revision = "e3a22ba34db5"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "guest_payment_incident",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("public_id", sa.String(length=36), nullable=False),
        sa.Column("guest_booking_id", sa.String(length=64), nullable=False),
        sa.Column("saferpay_transaction_id", sa.String(length=128), nullable=True),
        sa.Column("amount", sa.Numeric(12, 2), nullable=False),
        sa.Column(
            "currency",
            sa.String(length=3),
            server_default=sa.text("'CHF'"),
            nullable=False,
        ),
        sa.Column("financial_state", sa.String(length=16), nullable=False),
        sa.Column(
            "booking_ids_text",
            sa.Text(),
            server_default=sa.text("'[]'"),
            nullable=False,
        ),
        sa.Column(
            "recovery_count", sa.Integer(), server_default=sa.text("0"), nullable=False
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
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "financial_state IN ('uncertain', 'captured', 'failed', 'resolved')",
            name="ck_guest_payment_incident_state",
        ),
        sa.CheckConstraint(
            "currency = 'CHF'", name="ck_guest_payment_incident_currency"
        ),
        sa.CheckConstraint("amount >= 0", name="ck_guest_payment_incident_amount"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("public_id"),
        sa.UniqueConstraint("guest_booking_id"),
        sa.UniqueConstraint("saferpay_transaction_id"),
    )
    op.create_index(
        "ix_guest_payment_incident_state",
        "guest_payment_incident",
        ["financial_state"],
    )
    op.create_index(
        "ix_guest_payment_incident_created",
        "guest_payment_incident",
        ["created_at"],
    )
    op.create_table(
        "guest_payment_incident_event",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("incident_id", sa.Integer(), nullable=False),
        sa.Column("event_type", sa.String(length=32), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "payload_text", sa.Text(), server_default=sa.text("'{}'"), nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["incident_id"],
            ["guest_payment_incident.id"],
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(["actor_user_id"], ["user.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_guest_payment_incident_event_incident",
        "guest_payment_incident_event",
        ["incident_id"],
    )


def downgrade():
    op.drop_index(
        "ix_guest_payment_incident_event_incident",
        table_name="guest_payment_incident_event",
    )
    op.drop_table("guest_payment_incident_event")
    op.drop_index(
        "ix_guest_payment_incident_created", table_name="guest_payment_incident"
    )
    op.drop_index(
        "ix_guest_payment_incident_state", table_name="guest_payment_incident"
    )
    op.drop_table("guest_payment_incident")
