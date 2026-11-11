"""caisse plusieurs encaissements

Revision ID: 393ac737dd28
Revises: 8861d96930bb
Create Date: 2026-11-01 16:34:07.087684

Généré par Alembic, puis limité à cash_collection.
Le reste de l'autogénération reprenait un écart de modèles déjà présent
(index, payment, transport_requests, user) et ne fait pas partie de ce lot.
Les CHECK de kind et de net, absents du diff autogénéré, viennent du modèle.
"""

from alembic import op
import sqlalchemy as sa


revision = "393ac737dd28"
down_revision = "8861d96930bb"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "cash_collection",
        sa.Column(
            "kind",
            sa.String(length=32),
            server_default=sa.text("'collection'"),
            nullable=False,
        ),
    )
    op.add_column(
        "cash_collection",
        sa.Column("client_operation_key", sa.String(length=80), nullable=True),
    )
    op.execute(
        "UPDATE cash_collection "
        "SET client_operation_key = id::text "
        "WHERE client_operation_key IS NULL"
    )
    op.alter_column(
        "cash_collection",
        "client_operation_key",
        existing_type=sa.String(length=80),
        nullable=False,
        server_default=sa.text("gen_random_uuid()::text"),
    )
    op.add_column(
        "cash_collection",
        sa.Column("compensates_collection_id", sa.Integer(), nullable=True),
    )
    op.create_check_constraint(
        "ck_cash_collection_kind",
        "cash_collection",
        "kind IN ('collection', 'compensation')",
    )
    op.create_check_constraint(
        "ck_cash_collection_operation_key",
        "cash_collection",
        "char_length(client_operation_key) > 0",
    )
    op.create_check_constraint(
        "ck_cash_collection_kind_net",
        "cash_collection",
        "("
        "kind = 'collection' AND compensates_collection_id IS NULL "
        "AND (net_chf IS NULL OR net_chf >= 0)"
        ") OR ("
        "kind = 'compensation' AND compensates_collection_id IS NOT NULL "
        "AND compensates_collection_id <> id AND net_chf IS NOT NULL "
        "AND net_chf < 0"
        ")",
    )
    op.drop_constraint("uq_cash_collection_booking", "cash_collection", type_="unique")
    op.create_index(
        "ix_cash_collection_company_booking",
        "cash_collection",
        ["company_id", "booking_id"],
        unique=False,
    )
    op.create_unique_constraint(
        "uq_cash_collection_booking_operation",
        "cash_collection",
        ["booking_id", "client_operation_key"],
    )
    op.create_unique_constraint(
        "uq_cash_collection_scope",
        "cash_collection",
        ["id", "company_id", "booking_id"],
    )
    op.create_foreign_key(
        "fk_cash_collection_compensates_scope",
        "cash_collection",
        "cash_collection",
        ["compensates_collection_id", "company_id", "booking_id"],
        ["id", "company_id", "booking_id"],
        ondelete="RESTRICT",
    )


def downgrade():
    op.drop_constraint(
        "fk_cash_collection_compensates_scope",
        "cash_collection",
        type_="foreignkey",
    )
    op.drop_constraint("uq_cash_collection_scope", "cash_collection", type_="unique")
    op.drop_constraint(
        "uq_cash_collection_booking_operation",
        "cash_collection",
        type_="unique",
    )
    op.drop_index("ix_cash_collection_company_booking", table_name="cash_collection")
    op.create_unique_constraint(
        "uq_cash_collection_booking", "cash_collection", ["booking_id"]
    )
    op.drop_constraint("ck_cash_collection_kind_net", "cash_collection", type_="check")
    op.drop_constraint(
        "ck_cash_collection_operation_key", "cash_collection", type_="check"
    )
    op.drop_constraint("ck_cash_collection_kind", "cash_collection", type_="check")
    op.drop_column("cash_collection", "compensates_collection_id")
    op.drop_column("cash_collection", "client_operation_key")
    op.drop_column("cash_collection", "kind")
