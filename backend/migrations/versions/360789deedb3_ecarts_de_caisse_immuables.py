"""ecarts de caisse immuables

Revision ID: 360789deedb3
Revises: 25a6095d93ac
Create Date: 2026-11-01 21:45:00.000000

Généré par Alembic sur une base estampillée ``25a6095d93ac``,
puis limité à ``cash_reconciliation``. Le reste recréait des index déjà
présents dans la chaîne et ne fait pas partie de ce lot.

Les écarts déjà enregistrés deviennent des constats. Leur clé reprend
l'identifiant de la ligne. Une régularisation reste une nouvelle ligne.
"""

from alembic import op
import sqlalchemy as sa


revision = "360789deedb3"
down_revision = "25a6095d93ac"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "cash_reconciliation",
        sa.Column("collection_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "cash_reconciliation",
        sa.Column("driver_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "cash_reconciliation",
        sa.Column("reason", sa.String(length=400), nullable=True),
    )
    op.add_column(
        "cash_reconciliation",
        sa.Column("status", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "cash_reconciliation",
        sa.Column("idempotency_key", sa.String(length=80), nullable=True),
    )
    op.add_column(
        "cash_reconciliation",
        sa.Column("regularizes_id", sa.Integer(), nullable=True),
    )
    op.execute(
        "UPDATE cash_reconciliation AS line "
        "SET collection_id = handover.collection_id, "
        "driver_id = handover.driver_id, "
        "reason = 'rapprochement', "
        "status = 'constate', "
        "idempotency_key = 'legacy-' || line.id::text "
        "FROM cash_handover AS handover "
        "WHERE line.handover_id = handover.id "
        "AND line.idempotency_key IS NULL"
    )
    op.alter_column("cash_reconciliation", "collection_id", nullable=False)
    op.alter_column("cash_reconciliation", "driver_id", nullable=False)
    op.alter_column("cash_reconciliation", "reason", nullable=False)
    op.alter_column("cash_reconciliation", "status", nullable=False)
    op.alter_column("cash_reconciliation", "idempotency_key", nullable=False)
    op.drop_constraint(
        "uq_cash_reconciliation_handover", "cash_reconciliation", type_="unique"
    )
    op.create_index(
        "uq_cash_reconciliation_constat",
        "cash_reconciliation",
        ["handover_id"],
        unique=True,
        postgresql_where=sa.text("status = 'constate'"),
    )
    op.create_unique_constraint(
        "uq_cash_reconciliation_operation",
        "cash_reconciliation",
        ["company_id", "idempotency_key"],
    )
    op.create_foreign_key(
        "fk_cash_reconciliation_driver",
        "cash_reconciliation",
        "driver",
        ["driver_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_cash_reconciliation_collection",
        "cash_reconciliation",
        "cash_collection",
        ["collection_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_cash_reconciliation_regularizes",
        "cash_reconciliation",
        "cash_reconciliation",
        ["regularizes_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_check_constraint(
        "ck_cash_reconciliation_key",
        "cash_reconciliation",
        "char_length(idempotency_key) > 0",
    )
    op.create_check_constraint(
        "ck_cash_reconciliation_link",
        "cash_reconciliation",
        "(status = 'constate' AND regularizes_id IS NULL) OR "
        "(status = 'regularisation' AND regularizes_id IS NOT NULL)",
    )
    op.create_check_constraint(
        "ck_cash_reconciliation_reason",
        "cash_reconciliation",
        "char_length(reason) > 0",
    )
    op.create_check_constraint(
        "ck_cash_reconciliation_status",
        "cash_reconciliation",
        "status IN ('constate', 'regularisation')",
    )
    op.create_check_constraint(
        "ck_cash_reconciliation_variance",
        "cash_reconciliation",
        "variance_chf = (counted_chf - expected_chf)",
    )


def downgrade():
    op.drop_constraint(
        "ck_cash_reconciliation_variance", "cash_reconciliation", type_="check"
    )
    op.drop_constraint(
        "ck_cash_reconciliation_status", "cash_reconciliation", type_="check"
    )
    op.drop_constraint(
        "ck_cash_reconciliation_reason", "cash_reconciliation", type_="check"
    )
    op.drop_constraint(
        "ck_cash_reconciliation_link", "cash_reconciliation", type_="check"
    )
    op.drop_constraint(
        "ck_cash_reconciliation_key", "cash_reconciliation", type_="check"
    )
    op.drop_constraint(
        "fk_cash_reconciliation_regularizes", "cash_reconciliation", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_cash_reconciliation_collection", "cash_reconciliation", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_cash_reconciliation_driver", "cash_reconciliation", type_="foreignkey"
    )
    op.drop_constraint(
        "uq_cash_reconciliation_operation", "cash_reconciliation", type_="unique"
    )
    op.drop_index(
        "uq_cash_reconciliation_constat",
        table_name="cash_reconciliation",
        postgresql_where=sa.text("status = 'constate'"),
    )
    op.create_unique_constraint(
        "uq_cash_reconciliation_handover", "cash_reconciliation", ["handover_id"]
    )
    op.drop_column("cash_reconciliation", "regularizes_id")
    op.drop_column("cash_reconciliation", "idempotency_key")
    op.drop_column("cash_reconciliation", "status")
    op.drop_column("cash_reconciliation", "reason")
    op.drop_column("cash_reconciliation", "driver_id")
    op.drop_column("cash_reconciliation", "collection_id")
