"""f03_activation_delivery_supersession

Revision ID: b79c3a9a4958
Revises: d07b29c401ea
Create Date: 2026-07-26 20:32:24.718817

"""

from alembic import op
import sqlalchemy as sa


revision = "b79c3a9a4958"
down_revision = "d07b29c401ea"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("activation_email_deliveries", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("superseded_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("superseded_by_delivery_id", sa.String(length=36), nullable=True)
        )
        batch_op.create_index(
            "ix_act_email_del_session_superseded",
            ["activation_session_pk", "superseded_at"],
            unique=False,
        )

    # Backfill F-03 : courante ssi pointeur session == delivery_id ET même session
    op.execute(
        sa.text(
            """
            UPDATE activation_email_deliveries d
            SET superseded_at = NOW(),
                superseded_by_delivery_id = s.email_delivery_id,
                updated_at = NOW()
            FROM activation_session s
            WHERE d.activation_session_pk = s.id
              AND d.superseded_at IS NULL
              AND s.email_delivery_id IS NOT NULL
              AND EXISTS (
                SELECT 1 FROM activation_email_deliveries cur
                WHERE cur.email_delivery_id = s.email_delivery_id
                  AND cur.activation_session_pk = s.id
              )
              AND d.email_delivery_id <> s.email_delivery_id
            """
        )
    )
    # Pointeur absent / inexistant / inter-session → toutes superseded
    op.execute(
        sa.text(
            """
            UPDATE activation_email_deliveries d
            SET superseded_at = NOW(),
                superseded_by_delivery_id = NULL,
                updated_at = NOW()
            FROM activation_session s
            WHERE d.activation_session_pk = s.id
              AND d.superseded_at IS NULL
              AND (
                s.email_delivery_id IS NULL
                OR NOT EXISTS (
                  SELECT 1 FROM activation_email_deliveries cur
                  WHERE cur.email_delivery_id = s.email_delivery_id
                    AND cur.activation_session_pk = s.id
                )
              )
            """
        )
    )


def downgrade():
    with op.batch_alter_table("activation_email_deliveries", schema=None) as batch_op:
        batch_op.drop_index("ix_act_email_del_session_superseded")
        batch_op.drop_column("superseded_by_delivery_id")
        batch_op.drop_column("superseded_at")
