"""dossier fictif de reglement chauffeur

Revision ID: a2167da2d6ae
Revises: c4d8e71b05f2
Create Date: 2026-10-27 15:04:02.197722

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "a2167da2d6ae"
down_revision = "c4d8e71b05f2"
branch_labels = None
depends_on = None


def upgrade():
    # Généré par Alembic, puis limité aux deux tables du dossier fictif.
    # Le reste du diff d'autogénération reprenait un écart de modèles déjà
    # présent et ne fait pas partie de ce lot.
    op.create_table(
        "settlement_simulation_case",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=True),
        sa.Column("debtor_label", sa.String(length=160), nullable=False),
        sa.Column("amount_chf", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column(
            "currency",
            sa.String(length=3),
            server_default=sa.text("'CHF'"),
            nullable=False,
        ),
        sa.Column("mode", sa.String(length=20), nullable=True),
        sa.Column("decision_version", sa.String(length=64), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=True),
        sa.Column("attempt_id", sa.String(length=80), nullable=True),
        sa.Column(
            "state",
            sa.String(length=40),
            server_default=sa.text("'preparing'"),
            nullable=False,
        ),
        sa.Column(
            "contractual_basis",
            sa.String(length=40),
            server_default=sa.text("'missing'"),
            nullable=False,
        ),
        sa.Column(
            "signature_recorded",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "real_financial_write",
            sa.Boolean(),
            server_default=sa.text("false"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "currency = 'CHF'", name="ck_settlement_simulation_currency"
        ),
        sa.CheckConstraint(
            "mode IS NULL OR mode IN ('tap_to_pay', 'cash', 'invoice')",
            name="ck_settlement_simulation_mode",
        ),
        sa.CheckConstraint(
            "state IN ('preparing', 'operation_in_progress', "
            "'simulated_confirmed', 'simulated_refused', 'uncertain', "
            "'administrative_simulated', 'closed')",
            name="ck_settlement_simulation_state",
        ),
        sa.CheckConstraint(
            "real_financial_write = false",
            name="ck_settlement_simulation_not_real",
        ),
        sa.CheckConstraint(
            "signature_recorded = false",
            name="ck_settlement_simulation_no_signature",
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="RESTRICT"),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="RESTRICT"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("booking_id", name="uq_settlement_simulation_booking"),
    )
    op.create_table(
        "settlement_simulation_event",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("case_id", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=80), nullable=False),
        sa.Column("action", sa.String(length=40), nullable=False),
        sa.Column("from_state", sa.String(length=40), nullable=False),
        sa.Column("to_state", sa.String(length=40), nullable=False),
        sa.Column("actor_role", sa.String(length=20), nullable=False),
        sa.Column("actor_id", sa.Integer(), nullable=True),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["case_id"], ["settlement_simulation_case.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "case_id",
            "idempotency_key",
            name="uq_settlement_simulation_event_key",
        ),
    )
    op.create_index(
        "ix_settlement_simulation_event_case",
        "settlement_simulation_event",
        ["case_id", "id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_settlement_simulation_event_case",
        table_name="settlement_simulation_event",
    )
    op.drop_table("settlement_simulation_event")
    op.drop_table("settlement_simulation_case")
