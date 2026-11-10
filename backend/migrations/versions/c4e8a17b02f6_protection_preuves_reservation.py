"""protection des preuves de reservation

Revision ID: c4e8a17b02f6
Revises: b8d2f04c91aa
Create Date: 2026-10-25 20:50:00.000000

Empêche la suppression directe d'une réservation d'emporter un paiement,
une attestation ou le lien d'un aller-retour. Les autres cascades ne sont
pas modifiées. Aucune ligne existante n'est réécrite.
"""

from alembic import op


revision = "c4e8a17b02f6"
down_revision = "b8d2f04c91aa"
branch_labels = None
depends_on = None


def _replace(name: str, table: str, column: str, ondelete: str) -> None:
    op.drop_constraint(name, table, type_="foreignkey")
    op.create_foreign_key(
        name,
        table,
        "booking",
        [column],
        ["id"],
        ondelete=ondelete,
    )


def upgrade():
    _replace("payment_booking_id_fkey", "payment", "booking_id", "RESTRICT")
    _replace(
        "transport_attestation_booking_id_fkey",
        "transport_attestation",
        "booking_id",
        "RESTRICT",
    )
    _replace(
        "booking_parent_booking_id_fkey",
        "booking",
        "parent_booking_id",
        "RESTRICT",
    )
    _replace(
        "fk_booking_booking_group",
        "booking",
        "booking_group_id",
        "RESTRICT",
    )


def downgrade():
    _replace("payment_booking_id_fkey", "payment", "booking_id", "CASCADE")
    _replace(
        "transport_attestation_booking_id_fkey",
        "transport_attestation",
        "booking_id",
        "CASCADE",
    )
    _replace(
        "booking_parent_booking_id_fkey",
        "booking",
        "parent_booking_id",
        "SET NULL",
    )
    _replace(
        "fk_booking_booking_group",
        "booking",
        "booking_group_id",
        "SET NULL",
    )
