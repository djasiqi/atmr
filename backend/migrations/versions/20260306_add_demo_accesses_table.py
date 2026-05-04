"""add demo accesses table

Revision ID: 20260306_demo_accesses
Revises: 20260302_demo_requests
Create Date: 2026-03-06 10:15:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260306_demo_accesses"
down_revision = "20260302_demo_requests"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "demo_accesses",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("demo_request_id", sa.Integer(), nullable=False),
        sa.Column(
            "status", sa.String(length=32), nullable=False, server_default="pending"
        ),
        sa.Column("magic_token_hash", sa.String(length=128), nullable=True),
        sa.Column("magic_token_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("magic_token_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("demo_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("access_sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("provisioned_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expired_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("demo_user_id", sa.Integer(), nullable=True),
        sa.Column("demo_company_id", sa.Integer(), nullable=True),
        sa.Column("provision_source", sa.String(length=32), nullable=True),
        sa.Column("provisioning_mode", sa.String(length=32), nullable=True),
        sa.Column("last_access_email_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(
            ["demo_request_id"], ["demo_requests.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["demo_user_id"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["demo_company_id"], ["company.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_index(
        "ix_demo_accesses_status", "demo_accesses", ["status"], unique=False
    )
    op.create_index(
        "ix_demo_accesses_demo_expires_at",
        "demo_accesses",
        ["demo_expires_at"],
        unique=False,
    )
    op.create_index(
        "ix_demo_accesses_demo_request_id",
        "demo_accesses",
        ["demo_request_id"],
        unique=False,
    )
    op.create_index(
        "ix_demo_accesses_demo_request_created_at",
        "demo_accesses",
        ["demo_request_id", "created_at"],
        unique=False,
    )


def downgrade():
    op.drop_index(
        "ix_demo_accesses_demo_request_created_at", table_name="demo_accesses"
    )
    op.drop_index("ix_demo_accesses_demo_request_id", table_name="demo_accesses")
    op.drop_index("ix_demo_accesses_demo_expires_at", table_name="demo_accesses")
    op.drop_index("ix_demo_accesses_status", table_name="demo_accesses")
    op.drop_table("demo_accesses")
