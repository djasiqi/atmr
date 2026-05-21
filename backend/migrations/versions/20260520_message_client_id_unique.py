"""Index unique sender_id + client_message_id pour idempotence chat."""

from alembic import op

revision = "20260520_msg_idem_uq"
down_revision = "20260520_conversations"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("message", schema=None) as batch_op:
        batch_op.create_index(
            "uq_message_client_id_sender",
            ["sender_id", "client_message_id"],
            unique=True,
            postgresql_where="client_message_id IS NOT NULL",
        )


def downgrade():
    with op.batch_alter_table("message", schema=None) as batch_op:
        batch_op.drop_index("uq_message_client_id_sender")
