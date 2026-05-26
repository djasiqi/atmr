"""Add conversation engine tables and message.conversation_id.

Revision ID: 20260520_conversations
Revises: 20260519_msg_hub
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "20260520_conversations"
down_revision = "20260519_msg_hub"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "conversation",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=False),
        sa.Column("conversation_type", sa.String(length=32), nullable=False),
        sa.Column("context_type", sa.String(length=32), nullable=False),
        sa.Column("context_id", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("created_by", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("legacy_thread_id", sa.String(length=64), nullable=True),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(["company_id"], ["company.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["created_by"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conversation_company_type",
        "conversation",
        ["company_id", "conversation_type"],
    )
    op.create_index(
        "ix_conversation_context",
        "conversation",
        ["context_type", "context_id", "company_id"],
    )
    op.create_index(
        "ix_conversation_legacy_thread_id", "conversation", ["legacy_thread_id"]
    )

    op.create_table(
        "conversation_participant",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("conversation_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("participant_role", sa.String(length=32), nullable=False),
        sa.Column(
            "can_read", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column(
            "can_write", sa.Boolean(), nullable=False, server_default=sa.text("true")
        ),
        sa.Column(
            "can_manage", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column(
            "joined_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("left_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversation.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_conv_participant_user",
        "conversation_participant",
        ["user_id", "conversation_id"],
    )
    op.create_index(
        "uq_conversation_participant",
        "conversation_participant",
        ["conversation_id", "user_id"],
        unique=True,
    )

    op.create_table(
        "message_read",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("message_id", sa.Integer(), nullable=False),
        sa.Column(
            "read_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(["message_id"], ["message.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "uq_message_read_user_message",
        "message_read",
        ["user_id", "message_id"],
        unique=True,
    )

    with op.batch_alter_table("message", schema=None) as batch_op:
        batch_op.add_column(sa.Column("conversation_id", sa.Integer(), nullable=True))
        batch_op.add_column(
            sa.Column(
                "visibility_tags",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
            )
        )
        batch_op.add_column(
            sa.Column("system_event_key", sa.String(length=128), nullable=True)
        )
        batch_op.create_foreign_key(
            "fk_message_conversation_id",
            "conversation",
            ["conversation_id"],
            ["id"],
            ondelete="SET NULL",
        )
        batch_op.create_index(
            "ix_message_conversation_id", ["conversation_id"], unique=False
        )
        batch_op.create_index(
            "ix_message_system_event_key", ["system_event_key"], unique=False
        )


def downgrade():
    with op.batch_alter_table("message", schema=None) as batch_op:
        batch_op.drop_index("ix_message_system_event_key")
        batch_op.drop_index("ix_message_conversation_id")
        batch_op.drop_constraint("fk_message_conversation_id", type_="foreignkey")
        batch_op.drop_column("system_event_key")
        batch_op.drop_column("visibility_tags")
        batch_op.drop_column("conversation_id")

    op.drop_table("message_read")
    op.drop_index("uq_conversation_participant", table_name="conversation_participant")
    op.drop_index("ix_conv_participant_user", table_name="conversation_participant")
    op.drop_table("conversation_participant")
    op.drop_index("ix_conversation_legacy_thread_id", table_name="conversation")
    op.drop_index("ix_conversation_context", table_name="conversation")
    op.drop_index("ix_conversation_company_type", table_name="conversation")
    op.drop_table("conversation")
