"""restrict booking client fk

Revision ID: 3cdd9a610dbd
Revises: 20260419_scale_idx
Create Date: 2026-04-19 21:42:44.883942
"""

from alembic import op
import sqlalchemy as sa

revision = "3cdd9a610dbd"
down_revision = "20260419_scale_idx"
branch_labels = None
depends_on = None


def _assert_no_orphan_booking_clients() -> None:
    bind = op.get_bind()
    orphan_count = bind.execute(
        sa.text(
            """
            SELECT COUNT(*)
            FROM booking b
            LEFT JOIN client c ON c.id = b.client_id
            WHERE b.client_id IS NOT NULL
              AND c.id IS NULL
            """
        )
    ).scalar_one()
    if int(orphan_count or 0) > 0:
        raise RuntimeError(
            "Migration bloquée: booking.client_id contient des références orphelines."
        )


def upgrade():
    _assert_no_orphan_booking_clients()
    op.drop_constraint(op.f("booking_client_id_fkey"), "booking", type_="foreignkey")
    op.create_foreign_key(
        op.f("booking_client_id_fkey"),
        "booking",
        "client",
        ["client_id"],
        ["id"],
        ondelete="RESTRICT",
    )


def downgrade():
    op.drop_constraint(op.f("booking_client_id_fkey"), "booking", type_="foreignkey")
    op.create_foreign_key(
        op.f("booking_client_id_fkey"),
        "booking",
        "client",
        ["client_id"],
        ["id"],
        ondelete="CASCADE",
    )
