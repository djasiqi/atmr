"""activation_email_delivery_status

Revision ID: bd7f3de53566
Revises: f9b4b50f017d
Create Date: 2026-07-25 15:00:44.968473

"""

from alembic import op
import sqlalchemy as sa


revision = "bd7f3de53566"
down_revision = "f9b4b50f017d"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column(
        "activation_session",
        sa.Column("email_delivery_id", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "activation_session",
        sa.Column("email_delivery_status", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "activation_session",
        sa.Column("email_delivery_kind", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "activation_session",
        sa.Column("email_last_error", sa.Text(), nullable=True),
    )
    op.add_column(
        "activation_session",
        sa.Column("email_provider_message_id", sa.String(length=128), nullable=True),
    )
    op.create_index(
        "ix_activation_session_email_delivery_id",
        "activation_session",
        ["email_delivery_id"],
        unique=False,
    )

    # Backfill sûr : ne pas initialiser toutes les lignes à queued
    op.execute(
        """
        UPDATE activation_session
        SET email_delivery_status = 'sent',
            email_delivery_kind = COALESCE(email_delivery_kind, 'initial')
        WHERE email_verified_at IS NOT NULL
          AND email_delivery_status IS NULL
        """
    )
    op.execute(
        """
        UPDATE activation_session
        SET email_delivery_status = 'failed'
        WHERE email_verified_at IS NULL
          AND email_delivery_status IS NULL
        """
    )


def downgrade():
    op.drop_index(
        "ix_activation_session_email_delivery_id",
        table_name="activation_session",
    )
    op.drop_column("activation_session", "email_provider_message_id")
    op.drop_column("activation_session", "email_last_error")
    op.drop_column("activation_session", "email_delivery_kind")
    op.drop_column("activation_session", "email_delivery_status")
    op.drop_column("activation_session", "email_delivery_id")
