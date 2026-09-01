"""institution_billing_control_status

Revision ID: 453111f754df
Revises: 9c313d7f5a0a
Create Date: 2026-09-01 22:41:01.224799

INSTITUTION-07 — contrôle facturation institution (booking-level).
Autogenerate nettoyé : uniquement les colonnes cible sur ``booking``.
"""

from alembic import op
import sqlalchemy as sa

revision = "453111f754df"
down_revision = "9c313d7f5a0a"
branch_labels = None
depends_on = None

_CONTROL_STATUS = sa.Enum(
    "pending_review",
    "validated",
    "anomaly",
    name="institution_billing_control_status",
)


def upgrade():
    _CONTROL_STATUS.create(op.get_bind(), checkfirst=True)
    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "institution_control_status",
                _CONTROL_STATUS,
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "institution_control_validated_at",
                sa.DateTime(timezone=True),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "institution_control_validated_by_user_id",
                sa.Integer(),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column(
                "institution_control_validated_by_display_name",
                sa.String(length=200),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column("institution_control_anomaly_reason", sa.Text(), nullable=True)
        )
        batch_op.create_index(
            batch_op.f("ix_booking_institution_control_validated_by_user_id"),
            ["institution_control_validated_by_user_id"],
            unique=False,
        )
        batch_op.create_foreign_key(
            "fk_booking_institution_control_validated_by_user",
            "user",
            ["institution_control_validated_by_user_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade():
    with op.batch_alter_table("booking", schema=None) as batch_op:
        batch_op.drop_constraint(
            "fk_booking_institution_control_validated_by_user",
            type_="foreignkey",
        )
        batch_op.drop_index(
            batch_op.f("ix_booking_institution_control_validated_by_user_id")
        )
        batch_op.drop_column("institution_control_anomaly_reason")
        batch_op.drop_column("institution_control_validated_by_display_name")
        batch_op.drop_column("institution_control_validated_by_user_id")
        batch_op.drop_column("institution_control_validated_at")
        batch_op.drop_column("institution_control_status")
    _CONTROL_STATUS.drop(op.get_bind(), checkfirst=True)
