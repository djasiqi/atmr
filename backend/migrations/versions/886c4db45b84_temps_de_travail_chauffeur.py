"""temps de travail chauffeur

Revision ID: 886c4db45b84
Revises: b2ca98b6cb2d
Create Date: 2026-09-30 02:32:35.018595

Généré par ``flask db revision --autogenerate``. Les autres écarts détectés
(index et colonnes sans lien avec ce module) ont été retirés : les appliquer
aurait modifié des tables hors périmètre.
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "886c4db45b84"
down_revision = "b2ca98b6cb2d"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "driver_work_time_period_closure",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("period_from", sa.Date(), nullable=False),
        sa.Column("period_to", sa.Date(), nullable=False),
        sa.Column("finalized_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finalized_by_user_id", sa.Integer(), nullable=True),
        sa.Column("reopened_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reopened_by_user_id", sa.Integer(), nullable=True),
        sa.Column("reopen_reason", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["finalized_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["reopened_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_work_time_closure_period",
        "driver_work_time_period_closure",
        ["company_id", "period_from", "period_to"],
        unique=False,
    )

    op.create_table(
        "driver_compensation_ledger",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("closure_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=True),
        sa.Column("journey_key", sa.String(length=80), nullable=False),
        sa.Column("accounting_date", sa.Date(), nullable=True),
        sa.Column("compensated_minutes", sa.Integer(), nullable=False),
        sa.Column("policy_id", sa.Integer(), nullable=True),
        sa.Column("rule_type", sa.String(length=64), nullable=True),
        sa.Column("base_minutes", sa.Integer(), nullable=False),
        sa.Column("intermediate_stop_count", sa.Integer(), nullable=False),
        sa.Column("intermediate_stop_minutes", sa.Integer(), nullable=True),
        sa.Column("classification_source", sa.String(length=64), nullable=True),
        sa.Column("journey_status", sa.String(length=32), nullable=True),
        sa.Column("compensation_status", sa.String(length=32), nullable=True),
        sa.Column("attached_to_booking_id", sa.Integer(), nullable=True),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finalized_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["closure_id"], ["driver_work_time_period_closure.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_compensation_ledger_closure",
        "driver_compensation_ledger",
        ["closure_id", "journey_key"],
        unique=False,
    )
    op.create_index(
        "ix_driver_compensation_ledger_company_id",
        "driver_compensation_ledger",
        ["company_id"],
        unique=False,
    )

    op.create_table(
        "driver_compensation_policy",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=True),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("effective_until", sa.Date(), nullable=True),
        sa.Column("mode", sa.String(length=32), nullable=False),
        sa.Column("one_way_minutes", sa.Integer(), nullable=False),
        sa.Column("round_trip_minutes", sa.Integer(), nullable=False),
        sa.Column("intermediate_stop_minutes", sa.Integer(), nullable=True),
        sa.Column("max_reasonable_minutes", sa.Integer(), nullable=True),
        sa.Column(
            "overlap_threshold_minutes",
            sa.Integer(),
            server_default=sa.text("1"),
            nullable=False,
        ),
        sa.Column(
            "work_type_rules_json",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_driver_compensation_policy_company_id",
        "driver_compensation_policy",
        ["company_id"],
        unique=False,
    )
    op.create_index(
        "ix_driver_compensation_policy_driver_id",
        "driver_compensation_policy",
        ["driver_id"],
        unique=False,
    )
    op.create_index(
        "ix_driver_compensation_policy_scope",
        "driver_compensation_policy",
        ["company_id", "driver_id", "effective_from"],
        unique=False,
    )

    op.create_table(
        "driver_manual_work_entry",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("work_date", sa.Date(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("duration_minutes", sa.Integer(), nullable=False),
        sa.Column("work_type", sa.String(length=32), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("reference", sa.String(length=120), nullable=True),
        sa.Column("pickup_location", sa.String(length=500), nullable=True),
        sa.Column("dropoff_location", sa.String(length=500), nullable=True),
        sa.Column("booking_id", sa.Integer(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancelled_by_user_id", sa.Integer(), nullable=True),
        sa.Column("cancellation_reason", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["cancelled_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_manual_work_entry_driver_date",
        "driver_manual_work_entry",
        ["company_id", "driver_id", "work_date"],
        unique=False,
    )

    op.create_table(
        "driver_work_time_adjustment",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("original_arrived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("original_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("original_source", sa.String(length=64), nullable=True),
        sa.Column("corrected_arrived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("corrected_completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["booking_id"], ["booking.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["driver_id"], ["driver.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_driver_work_time_adjustment_company_id",
        "driver_work_time_adjustment",
        ["company_id"],
        unique=False,
    )
    op.create_index(
        "ix_driver_work_time_adjustment_driver_id",
        "driver_work_time_adjustment",
        ["driver_id"],
        unique=False,
    )
    op.create_index(
        "ix_work_time_adjustment_booking",
        "driver_work_time_adjustment",
        ["booking_id", "created_at"],
        unique=False,
    )

    op.add_column(
        "booking",
        sa.Column("arrived_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_booking_driver_completed_at",
        "booking",
        ["driver_id", "completed_at"],
        unique=False,
    )
    op.create_index(
        "ix_booking_driver_scheduled_missing_completed",
        "booking",
        ["driver_id", "scheduled_time"],
        unique=False,
        postgresql_where=sa.text("completed_at IS NULL"),
    )


def downgrade():
    op.drop_index(
        "ix_booking_driver_scheduled_missing_completed",
        table_name="booking",
    )
    op.drop_index("ix_booking_driver_completed_at", table_name="booking")
    op.drop_column("booking", "arrived_at")
    op.drop_index(
        "ix_work_time_adjustment_booking", table_name="driver_work_time_adjustment"
    )
    op.drop_index(
        "ix_driver_work_time_adjustment_driver_id",
        table_name="driver_work_time_adjustment",
    )
    op.drop_index(
        "ix_driver_work_time_adjustment_company_id",
        table_name="driver_work_time_adjustment",
    )
    op.drop_table("driver_work_time_adjustment")
    op.drop_index(
        "ix_manual_work_entry_driver_date", table_name="driver_manual_work_entry"
    )
    op.drop_table("driver_manual_work_entry")
    op.drop_index(
        "ix_driver_compensation_policy_scope", table_name="driver_compensation_policy"
    )
    op.drop_index(
        "ix_driver_compensation_policy_driver_id",
        table_name="driver_compensation_policy",
    )
    op.drop_index(
        "ix_driver_compensation_policy_company_id",
        table_name="driver_compensation_policy",
    )
    op.drop_table("driver_compensation_policy")
    op.drop_index(
        "ix_driver_compensation_ledger_company_id",
        table_name="driver_compensation_ledger",
    )
    op.drop_index(
        "ix_compensation_ledger_closure", table_name="driver_compensation_ledger"
    )
    op.drop_table("driver_compensation_ledger")
    op.drop_index(
        "ix_work_time_closure_period", table_name="driver_work_time_period_closure"
    )
    op.drop_table("driver_work_time_period_closure")
