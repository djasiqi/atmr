"""add cancellation fields to booking

Revision ID: 20260202_cancellation_fields
Revises: 20260201_s2_exclude_cancelled
Create Date: 2026-02-02

Colonnes pour annulation standardisée : motif obligatoire, facturation déterministe.
Backfill : bookings annulés existants → OTHER, non facturé, libellé historique.
"""
from alembic import op
import sqlalchemy as sa


revision = "20260202_cancellation_fields"
down_revision = "20260201_s2_exclude_cancelled"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.add_column(sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("cancelled_by_role", sa.String(20), nullable=True))
        batch_op.add_column(sa.Column("cancellation_reason_code", sa.String(50), nullable=True))
        batch_op.add_column(sa.Column("cancellation_reason_text", sa.Text(), nullable=True))
        batch_op.add_column(sa.Column("is_cancellation_billable", sa.Boolean(), nullable=True))
        batch_op.add_column(sa.Column("cancellation_display_label", sa.String(120), nullable=True))

    # Backfill : bookings annulés sans motif → historique
    # Note: l'enum booking_status ne contient que 'CANCELED' (US spelling)
    op.execute(
        sa.text("""
            UPDATE booking
            SET
                cancellation_reason_code = 'OTHER',
                is_cancellation_billable = false,
                cancellation_display_label = 'Annulation (historique)'
            WHERE status = 'CANCELED'
              AND cancellation_reason_code IS NULL
        """)
    )


def downgrade():
    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.drop_column("cancellation_display_label")
        batch_op.drop_column("is_cancellation_billable")
        batch_op.drop_column("cancellation_reason_text")
        batch_op.drop_column("cancellation_reason_code")
        batch_op.drop_column("cancelled_by_role")
        batch_op.drop_column("cancelled_at")
