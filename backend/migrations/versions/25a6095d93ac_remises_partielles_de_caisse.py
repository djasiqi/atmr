"""remises partielles de caisse

Revision ID: 25a6095d93ac
Revises: acce98cae5db
Create Date: 2026-11-01 21:20:00.000000

Généré par Alembic sur une base estampillée ``acce98cae5db``,
puis limité à ``cash_handover``. Le reste recréait des index déjà
présents dans la chaîne et ne fait pas partie de ce lot.

La clé est d'abord remplie pour les remises historiques, puis rendue
obligatoire. L'unicité porte sur la remise, plus sur la collection.
"""

from alembic import op
import sqlalchemy as sa


revision = "25a6095d93ac"
down_revision = "acce98cae5db"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "cash_handover",
        sa.Column("idempotency_key", sa.String(length=80), nullable=True),
    )
    op.execute(
        "UPDATE cash_handover "
        "SET idempotency_key = 'legacy-' || id::text "
        "WHERE idempotency_key IS NULL"
    )
    op.alter_column("cash_handover", "idempotency_key", nullable=False)
    op.drop_constraint("uq_cash_handover_collection", "cash_handover", type_="unique")
    op.create_unique_constraint(
        "uq_cash_handover_operation",
        "cash_handover",
        ["collection_id", "idempotency_key"],
    )
    op.create_check_constraint(
        "ck_cash_handover_amount", "cash_handover", "amount_chf > 0"
    )
    op.create_check_constraint(
        "ck_cash_handover_key", "cash_handover", "char_length(idempotency_key) > 0"
    )


def downgrade():
    op.drop_constraint("ck_cash_handover_key", "cash_handover", type_="check")
    op.drop_constraint("ck_cash_handover_amount", "cash_handover", type_="check")
    op.drop_constraint("uq_cash_handover_operation", "cash_handover", type_="unique")
    op.create_unique_constraint(
        "uq_cash_handover_collection", "cash_handover", ["collection_id"]
    )
    op.drop_column("cash_handover", "idempotency_key")
