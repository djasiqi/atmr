"""web_session_institution_idle_p0a

Revision ID: 861b5acea642
Revises: 25ce766952e2
Create Date: 2026-08-23 13:58:03.750281

Migration chirurgicale : table web_session + colonne refresh_token.web_session_id.
"""

from alembic import op
import sqlalchemy as sa


revision = "861b5acea642"
down_revision = "25ce766952e2"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "web_session",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("institution_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "last_interactive_activity_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_reason", sa.String(length=255), nullable=True),
        sa.Column("ip_address", sa.String(length=45), nullable=True),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["institution_id"], ["institutions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_web_session_institution_id", "web_session", ["institution_id"])
    op.create_index("ix_web_session_revoked_at", "web_session", ["revoked_at"])
    op.create_index("ix_web_session_user_id", "web_session", ["user_id"])

    with op.batch_alter_table("refresh_token", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column("web_session_id", sa.String(length=36), nullable=True)
        )
        batch_op.create_index(
            batch_op.f("ix_refresh_token_web_session_id"),
            ["web_session_id"],
            unique=False,
        )
        batch_op.create_foreign_key(
            "fk_refresh_token_web_session_id",
            "web_session",
            ["web_session_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade():
    with op.batch_alter_table("refresh_token", schema=None) as batch_op:
        batch_op.drop_constraint("fk_refresh_token_web_session_id", type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_refresh_token_web_session_id"))
        batch_op.drop_column("web_session_id")

    op.drop_index("ix_web_session_user_id", table_name="web_session")
    op.drop_index("ix_web_session_revoked_at", table_name="web_session")
    op.drop_index("ix_web_session_institution_id", table_name="web_session")
    op.drop_table("web_session")
