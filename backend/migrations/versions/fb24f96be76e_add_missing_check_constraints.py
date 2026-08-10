"""add_missing_check_constraints

Revision ID: fb24f96be76e
Revises: 4490a30d6a68
Create Date: 2026-08-10 02:32:29.065010

Autogenerate (comments billing_origin*) + contraintes CHECK présentes
dans les modèles mais absentes de la DB après upgrade heads
(chk_booking_assigned_requires_driver, chk_invoice_balance_nonneg,
chk_invoice_paid_nonneg). Pré-check données : 0 violation locale.
"""

from alembic import op
import sqlalchemy as sa


revision = "fb24f96be76e"
down_revision = "4490a30d6a68"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.alter_column(
            "billing_origin",
            existing_type=sa.VARCHAR(length=32),
            comment="OWN_PORTFOLIO | LIRIE_MARKETPLACE | IMPORTED | ADMIN_CREATED | UNKNOWN",
            existing_nullable=True,
        )
        batch_op.alter_column(
            "billing_origin_source",
            existing_type=sa.VARCHAR(length=32),
            comment="EXPLICIT_AT_CREATION | BACKFILL_* | ADMIN_CORRECTION",
            existing_nullable=True,
        )
        batch_op.create_check_constraint(
            "chk_booking_assigned_requires_driver",
            "status != 'ASSIGNED' OR driver_id IS NOT NULL",
        )

    with op.batch_alter_table("invoices", schema=None) as batch_op:
        batch_op.create_check_constraint(
            "chk_invoice_balance_nonneg",
            "balance_due >= 0",
        )
        batch_op.create_check_constraint(
            "chk_invoice_paid_nonneg",
            "amount_paid >= 0",
        )


def downgrade():
    with op.batch_alter_table("invoices", schema=None) as batch_op:
        batch_op.drop_constraint("chk_invoice_paid_nonneg", type_="check")
        batch_op.drop_constraint("chk_invoice_balance_nonneg", type_="check")

    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.drop_constraint("chk_booking_assigned_requires_driver", type_="check")
        batch_op.alter_column(
            "billing_origin_source",
            existing_type=sa.VARCHAR(length=32),
            comment=None,
            existing_comment="EXPLICIT_AT_CREATION | BACKFILL_* | ADMIN_CORRECTION",
            existing_nullable=True,
        )
        batch_op.alter_column(
            "billing_origin",
            existing_type=sa.VARCHAR(length=32),
            comment=None,
            existing_comment=(
                "OWN_PORTFOLIO | LIRIE_MARKETPLACE | IMPORTED | ADMIN_CREATED | UNKNOWN"
            ),
            existing_nullable=True,
        )
