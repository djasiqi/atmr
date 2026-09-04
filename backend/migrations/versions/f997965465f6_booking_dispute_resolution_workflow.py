"""booking_dispute_resolution_workflow

Revision ID: f997965465f6
Revises: 453111f754df
Create Date: 2026-09-04 16:15:38.728232

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "f997965465f6"
down_revision = "453111f754df"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "booking_disputes",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=True),
        sa.Column("institution_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=40), nullable=False),
        sa.Column(
            "opened_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("opened_by_user_id", sa.Integer(), nullable=True),
        sa.Column("institution_reason_code", sa.String(length=64), nullable=True),
        sa.Column("institution_reason_text", sa.Text(), nullable=True),
        sa.Column("frozen_amount_ht", sa.Numeric(precision=12, scale=2), nullable=True),
        sa.Column("frozen_payer_type", sa.String(length=32), nullable=True),
        sa.Column("frozen_billing_party_id", sa.Integer(), nullable=True),
        sa.Column("carrier_stance", sa.String(length=40), nullable=True),
        sa.Column("carrier_exclusion_reason", sa.String(length=64), nullable=True),
        sa.Column("carrier_note", sa.Text(), nullable=True),
        sa.Column("carrier_responded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("carrier_responded_by_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "proposed_amount_ht", sa.Numeric(precision=12, scale=2), nullable=True
        ),
        sa.Column("proposed_payer_type", sa.String(length=32), nullable=True),
        sa.Column("proposed_correction_note", sa.Text(), nullable=True),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by_user_id", sa.Integer(), nullable=True),
        sa.Column("resolver_role", sa.String(length=40), nullable=True),
        sa.Column("resolution_note", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(
            ["carrier_responded_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["institution_id"], ["institutions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["opened_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["resolved_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_booking_disputes_booking_id"),
        "booking_disputes",
        ["booking_id"],
        unique=False,
    )
    op.create_index(
        "ix_booking_disputes_booking_status",
        "booking_disputes",
        ["booking_id", "status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_booking_disputes_company_id"),
        "booking_disputes",
        ["company_id"],
        unique=False,
    )
    op.create_index(
        "ix_booking_disputes_company_status",
        "booking_disputes",
        ["company_id", "status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_booking_disputes_institution_id"),
        "booking_disputes",
        ["institution_id"],
        unique=False,
    )
    op.create_index(
        "ix_booking_disputes_institution_status",
        "booking_disputes",
        ["institution_id", "status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_booking_disputes_status"),
        "booking_disputes",
        ["status"],
        unique=False,
    )
    op.create_table(
        "booking_dispute_events",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("dispute_id", sa.BigInteger(), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("actor_user_id", sa.Integer(), nullable=True),
        sa.Column("actor_role", sa.String(length=40), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["actor_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["dispute_id"], ["booking_disputes.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_booking_dispute_events_dispute_id"),
        "booking_dispute_events",
        ["dispute_id"],
        unique=False,
    )
    op.create_table(
        "booking_dispute_evidence",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("dispute_id", sa.BigInteger(), nullable=False),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("stored_path", sa.String(length=512), nullable=True),
        sa.Column("original_filename", sa.String(length=255), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["dispute_id"], ["booking_disputes.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_booking_dispute_evidence_dispute_id"),
        "booking_dispute_evidence",
        ["dispute_id"],
        unique=False,
    )
    op.add_column(
        "booking",
        sa.Column("invoice_billing_status", sa.String(length=32), nullable=True),
    )


def downgrade():
    op.drop_column("booking", "invoice_billing_status")
    op.drop_index(
        op.f("ix_booking_dispute_evidence_dispute_id"),
        table_name="booking_dispute_evidence",
    )
    op.drop_table("booking_dispute_evidence")
    op.drop_index(
        op.f("ix_booking_dispute_events_dispute_id"),
        table_name="booking_dispute_events",
    )
    op.drop_table("booking_dispute_events")
    op.drop_index(op.f("ix_booking_disputes_status"), table_name="booking_disputes")
    op.drop_index(
        "ix_booking_disputes_institution_status", table_name="booking_disputes"
    )
    op.drop_index(
        op.f("ix_booking_disputes_institution_id"), table_name="booking_disputes"
    )
    op.drop_index("ix_booking_disputes_company_status", table_name="booking_disputes")
    op.drop_index(op.f("ix_booking_disputes_company_id"), table_name="booking_disputes")
    op.drop_index("ix_booking_disputes_booking_status", table_name="booking_disputes")
    op.drop_index(op.f("ix_booking_disputes_booking_id"), table_name="booking_disputes")
    op.drop_table("booking_disputes")
