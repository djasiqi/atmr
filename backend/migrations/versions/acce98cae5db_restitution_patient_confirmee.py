"""restitution patient confirmee

Revision ID: acce98cae5db
Revises: 152ed47ad70b
Create Date: 2026-11-01 20:30:00.000000

Généré par Alembic sur une base estampillée ``152ed47ad70b``,
puis limité aux colonnes de ``invoice_refund``. Le reste recréait des
index déjà présents dans la chaîne et ne fait pas partie de ce lot.
"""

from alembic import op
import sqlalchemy as sa


revision = "acce98cae5db"
down_revision = "152ed47ad70b"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "invoice_refund", sa.Column("booking_id", sa.Integer(), nullable=True)
    )
    op.add_column("invoice_refund", sa.Column("reason", sa.Text(), nullable=False))
    op.add_column(
        "invoice_refund", sa.Column("nature", sa.String(length=40), nullable=False)
    )
    op.add_column(
        "invoice_refund",
        sa.Column(
            "status",
            sa.String(length=20),
            server_default=sa.text("'requested'"),
            nullable=False,
        ),
    )
    op.add_column(
        "invoice_refund", sa.Column("created_by_user_id", sa.Integer(), nullable=False)
    )
    op.add_column(
        "invoice_refund",
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.alter_column(
        "invoice_refund",
        "invoice_id",
        existing_type=sa.INTEGER(),
        nullable=True,
    )
    op.create_index(
        "ix_invoice_refund_invoice",
        "invoice_refund",
        ["company_id", "invoice_id"],
        unique=False,
    )
    op.create_foreign_key(
        "fk_invoice_refund_booking",
        "invoice_refund",
        "booking",
        ["booking_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_invoice_refund_created_by",
        "invoice_refund",
        "user",
        ["created_by_user_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_check_constraint(
        "ck_invoice_refund_confirmed_at",
        "invoice_refund",
        "(status = 'confirmed' AND confirmed_at IS NOT NULL) "
        "OR (status <> 'confirmed' AND confirmed_at IS NULL)",
    )
    op.create_check_constraint(
        "ck_invoice_refund_nature",
        "invoice_refund",
        "nature IN ('restitution_especes', 'restitution_facture')",
    )
    op.create_check_constraint(
        "ck_invoice_refund_reason",
        "invoice_refund",
        "char_length(reason) > 0",
    )
    op.create_check_constraint(
        "ck_invoice_refund_status",
        "invoice_refund",
        "status IN ('requested', 'authorized', 'confirmed', 'refused', 'cancelled')",
    )
    op.create_check_constraint(
        "ck_invoice_refund_target",
        "invoice_refund",
        "(invoice_id IS NOT NULL) OR (cash_collection_id IS NOT NULL)",
    )


def downgrade():
    op.drop_constraint("ck_invoice_refund_target", "invoice_refund", type_="check")
    op.drop_constraint("ck_invoice_refund_status", "invoice_refund", type_="check")
    op.drop_constraint("ck_invoice_refund_reason", "invoice_refund", type_="check")
    op.drop_constraint("ck_invoice_refund_nature", "invoice_refund", type_="check")
    op.drop_constraint(
        "ck_invoice_refund_confirmed_at", "invoice_refund", type_="check"
    )
    op.drop_constraint(
        "fk_invoice_refund_created_by", "invoice_refund", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_invoice_refund_booking", "invoice_refund", type_="foreignkey"
    )
    op.drop_index("ix_invoice_refund_invoice", table_name="invoice_refund")
    op.alter_column(
        "invoice_refund",
        "invoice_id",
        existing_type=sa.INTEGER(),
        nullable=False,
    )
    op.drop_column("invoice_refund", "confirmed_at")
    op.drop_column("invoice_refund", "created_by_user_id")
    op.drop_column("invoice_refund", "status")
    op.drop_column("invoice_refund", "nature")
    op.drop_column("invoice_refund", "reason")
    op.drop_column("invoice_refund", "booking_id")
