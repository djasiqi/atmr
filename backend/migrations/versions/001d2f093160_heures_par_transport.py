"""heures par transport

Revision ID: 001d2f093160
Revises: 9ba102797f17
Create Date: 2026-09-30 22:07:24.896275

"""

from alembic import op
import sqlalchemy as sa


revision = "001d2f093160"
down_revision = "9ba102797f17"
branch_labels = None
depends_on = None


def upgrade():
    # Conservé depuis l'autogénération : uniquement le temps de travail.
    op.add_column(
        "driver_compensation_policy",
        sa.Column("transport_flat_minutes", sa.Integer(), nullable=True),
    )
    op.execute(
        "UPDATE driver_compensation_policy "
        "SET transport_flat_minutes = one_way_minutes "
        "WHERE transport_flat_minutes IS NULL"
    )
    op.alter_column(
        "driver_compensation_policy",
        "transport_flat_minutes",
        existing_type=sa.Integer(),
        nullable=False,
    )
    op.add_column(
        "driver_compensation_ledger",
        sa.Column("line_key", sa.String(length=80), nullable=True),
    )
    op.add_column(
        "driver_compensation_ledger",
        sa.Column("booking_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "driver_compensation_ledger",
        sa.Column("manual_entry_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "driver_compensation_ledger",
        sa.Column("flat_minutes", sa.Integer(), nullable=True),
    )
    op.execute(
        "UPDATE driver_compensation_ledger "
        "SET line_key = 'legacy:' || id::text "
        "WHERE line_key IS NULL"
    )
    op.alter_column(
        "driver_compensation_ledger",
        "line_key",
        existing_type=sa.String(length=80),
        nullable=False,
    )
    op.drop_index(
        "ix_compensation_ledger_closure", table_name="driver_compensation_ledger"
    )
    op.create_index(
        "uq_compensation_ledger_line",
        "driver_compensation_ledger",
        ["closure_id", "line_key"],
        unique=True,
    )


def downgrade():
    op.drop_index(
        "uq_compensation_ledger_line", table_name="driver_compensation_ledger"
    )
    op.create_index(
        "ix_compensation_ledger_closure",
        "driver_compensation_ledger",
        ["closure_id", "journey_key"],
        unique=False,
    )
    op.drop_column("driver_compensation_ledger", "flat_minutes")
    op.drop_column("driver_compensation_ledger", "manual_entry_id")
    op.drop_column("driver_compensation_ledger", "booking_id")
    op.drop_column("driver_compensation_ledger", "line_key")
    op.drop_column("driver_compensation_policy", "transport_flat_minutes")
