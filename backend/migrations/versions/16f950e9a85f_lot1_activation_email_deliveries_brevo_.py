"""lot1_activation_email_deliveries_brevo_webhook_events

Revision ID: 16f950e9a85f
Revises: d5e99a828da1
Create Date: 2026-07-26 12:25:32.525574

"""

from alembic import op
import sqlalchemy as sa


revision = "16f950e9a85f"
down_revision = "d5e99a828da1"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "brevo_webhook_events",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=64), nullable=False),
        sa.Column("event_type", sa.String(length=64), nullable=True),
        sa.Column("provider_message_id", sa.String(length=128), nullable=True),
        sa.Column("email_delivery_id", sa.String(length=36), nullable=True),
        sa.Column(
            "processed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("idempotency_key"),
    )
    op.create_table(
        "activation_email_deliveries",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("activation_session_pk", sa.Integer(), nullable=False),
        sa.Column("email_delivery_id", sa.String(length=36), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "token_key_version",
            sa.Integer(),
            server_default="1",
            nullable=False,
        ),
        sa.Column("email_token_hash", sa.String(length=64), nullable=True),
        sa.Column("token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provider_message_id", sa.String(length=128), nullable=True),
        sa.Column("provider_accepted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sending_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["activation_session_pk"],
            ["activation_session.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("email_delivery_id"),
    )
    with op.batch_alter_table("activation_email_deliveries", schema=None) as batch_op:
        batch_op.create_index(
            "ix_act_email_del_delivery_id", ["email_delivery_id"], unique=True
        )
        batch_op.create_index(
            "ix_act_email_del_provider_msg", ["provider_message_id"], unique=False
        )
        batch_op.create_index(
            "ix_act_email_del_session_id", ["activation_session_pk"], unique=False
        )
        batch_op.create_index(
            "ix_act_email_del_token_hash", ["email_token_hash"], unique=False
        )


def downgrade():
    with op.batch_alter_table("activation_email_deliveries", schema=None) as batch_op:
        batch_op.drop_index("ix_act_email_del_token_hash")
        batch_op.drop_index("ix_act_email_del_session_id")
        batch_op.drop_index("ix_act_email_del_provider_msg")
        batch_op.drop_index("ix_act_email_del_delivery_id")
    op.drop_table("activation_email_deliveries")
    op.drop_table("brevo_webhook_events")
