"""portal receivable dispute and hold index

Revision ID: 919313ab962c
Revises: 75471adb419d
Create Date: 2026-09-23 23:15:26.145701

Tables / index générés par ``flask db revision --autogenerate``.
Écarts hors périmètre retirés. Pas de ``batch_alter_table``.
"""

import sqlalchemy as sa
from alembic import op

revision = "919313ab962c"
down_revision = "75471adb419d"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "portal_receivable_dispute",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("receivable_id", sa.Integer(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("disputed_by_user_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by_user_id", sa.Integer(), nullable=True),
        sa.Column("resolution_note", sa.Text(), nullable=True),
        sa.CheckConstraint(
            "status IN ('open', 'accepted', 'rejected')",
            name="ck_portal_receivable_dispute_status",
        ),
        sa.ForeignKeyConstraint(
            ["disputed_by_user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["receivable_id"], ["portal_receivable.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["resolved_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_portal_receivable_dispute_receivable_id",
        "portal_receivable_dispute",
        ["receivable_id"],
        unique=False,
    )
    op.create_index(
        "ix_portal_receivable_dispute_open",
        "portal_receivable_dispute",
        ["receivable_id"],
        unique=True,
        postgresql_where=sa.text("status = 'open'"),
    )
    op.create_index(
        "ix_portal_receivable_debtor_creditor",
        "portal_receivable",
        ["debtor_user_id", "creditor_company_id"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_portal_receivable_debtor_creditor",
        table_name="portal_receivable",
    )
    op.drop_index(
        "ix_portal_receivable_dispute_open",
        table_name="portal_receivable_dispute",
    )
    op.drop_index(
        "ix_portal_receivable_dispute_receivable_id",
        table_name="portal_receivable_dispute",
    )
    op.drop_table("portal_receivable_dispute")
