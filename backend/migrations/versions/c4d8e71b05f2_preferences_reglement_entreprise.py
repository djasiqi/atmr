"""Préférences entreprise, distinctes des moyens réellement disponibles.

Revision ciblée, enfant de b3e7a91c04d2. Réservée à atmr_test.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c4d8e71b05f2"
down_revision = "b3e7a91c04d2"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "company_settlement_capability",
        sa.Column(
            "tap_to_pay_preferred",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.add_column(
        "company_settlement_capability",
        sa.Column(
            "cash_preferred",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.add_column(
        "company_settlement_capability",
        sa.Column(
            "direct_patient_invoice_preferred",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
    )
    op.create_table(
        "company_settlement_capability_revision",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column(
            "tap_to_pay_preferred",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "cash_preferred",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "direct_patient_invoice_preferred",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column("author_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["author_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.UniqueConstraint(
            "company_id",
            "revision",
            name="uq_company_settlement_capability_revision",
        ),
    )
    op.create_table(
        "booking_settlement_patient_notice",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("amount_chf", sa.Numeric(10, 2), nullable=True),
        sa.Column(
            "presented_modes",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("conditions_version", sa.String(64), nullable=False),
        sa.Column("author_user_id", sa.Integer(), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("channel", sa.String(32), nullable=False),
        sa.Column(
            "receipt_proven",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "receipt_proven = false",
            name="ck_patient_notice_not_proof",
        ),
        sa.ForeignKeyConstraint(
            ["booking_id"], ["booking.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["company_id"], ["company.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["author_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
    )


def downgrade():
    op.drop_table("booking_settlement_patient_notice")
    op.drop_table("company_settlement_capability_revision")
    op.drop_column("company_settlement_capability", "direct_patient_invoice_preferred")
    op.drop_column("company_settlement_capability", "cash_preferred")
    op.drop_column("company_settlement_capability", "tap_to_pay_preferred")
