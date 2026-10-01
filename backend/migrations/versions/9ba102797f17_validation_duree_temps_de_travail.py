"""validation duree temps de travail

Revision ID: 9ba102797f17
Revises: 886c4db45b84
Create Date: 2026-09-30 10:56:21.688696

"""

from alembic import op
import sqlalchemy as sa


revision = "9ba102797f17"
down_revision = "886c4db45b84"
branch_labels = None
depends_on = None


def upgrade():
    # Conservé depuis l'autogénération : uniquement ce module.
    # Les autres écarts de schéma détectés n'appartiennent pas à cette révision.
    op.create_table(
        "driver_work_time_settings",
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column(
            "route_margin_minutes",
            sa.Integer(),
            server_default=sa.text("5"),
            nullable=False,
        ),
        sa.Column(
            "route_estimate_enabled",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("company_id"),
    )
    op.create_table(
        "driver_work_time_duration_decision",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("driver_id", sa.Integer(), nullable=False),
        sa.Column("booking_id", sa.Integer(), nullable=False),
        sa.Column("proposed_worked_minutes", sa.Integer(), nullable=False),
        sa.Column("validated_worked_minutes", sa.Integer(), nullable=False),
        sa.Column("route_minutes", sa.Integer(), nullable=True),
        sa.Column("margin_minutes", sa.Integer(), nullable=True),
        sa.Column("route_provider", sa.String(length=32), nullable=True),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
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
        op.f("ix_driver_work_time_duration_decision_company_id"),
        "driver_work_time_duration_decision",
        ["company_id"],
        unique=False,
    )
    op.create_index(
        "ix_work_time_duration_decision_booking",
        "driver_work_time_duration_decision",
        ["booking_id", "created_at"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_work_time_duration_decision_booking",
        table_name="driver_work_time_duration_decision",
    )
    op.drop_index(
        op.f("ix_driver_work_time_duration_decision_company_id"),
        table_name="driver_work_time_duration_decision",
    )
    op.drop_table("driver_work_time_duration_decision")
    op.drop_table("driver_work_time_settings")
