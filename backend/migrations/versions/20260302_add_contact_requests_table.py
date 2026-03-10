"""add contact requests table

Revision ID: 20260302_contact_requests
Revises: 20260302_geo_unit_geom
Create Date: 2026-03-02 16:20:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = "20260302_contact_requests"
down_revision = "20260302_geo_unit_geom"
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    inspector = inspect(bind)
    has_users_table = "users" in inspector.get_table_names()

    op.create_table(
        "contact_requests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("organization", sa.String(length=180), nullable=True),
        sa.Column("phone", sa.String(length=32), nullable=True),
        sa.Column("category", sa.String(length=32), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("message_normalized", sa.Text(), nullable=True),
        sa.Column("dedupe_hash", sa.String(length=64), nullable=True),
        sa.Column("dedupe_window_bucket", sa.DateTime(timezone=True), nullable=True),
        sa.Column("client_request_id", sa.String(length=64), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("ip_hash", sa.String(length=128), nullable=True),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("user_public_id", sa.String(length=64), nullable=True),
        sa.Column("user_role", sa.String(length=32), nullable=True),
        sa.Column("company_id", sa.Integer(), nullable=True),
        sa.Column("institution_id", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(length=32), server_default="new", nullable=False),
        sa.Column("priority", sa.String(length=16), server_default="standard", nullable=False),
        sa.Column("assigned_channel", sa.String(length=120), nullable=True),
        sa.Column("email_delivery_status", sa.String(length=32), server_default="pending", nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    if has_users_table:
        op.create_foreign_key(
            "fk_contact_requests_user_id_users",
            "contact_requests",
            "users",
            ["user_id"],
            ["id"],
            ondelete="SET NULL",
        )
    op.create_index("ix_contact_requests_created_at", "contact_requests", ["created_at"], unique=False)
    op.create_index(
        "ix_contact_requests_category_status",
        "contact_requests",
        ["category", "status"],
        unique=False,
    )
    op.create_index("ix_contact_requests_email", "contact_requests", ["email"], unique=False)
    op.create_index("ix_contact_requests_trace_id", "contact_requests", ["trace_id"], unique=False)
    op.create_index("ix_contact_requests_user_id", "contact_requests", ["user_id"], unique=False)
    op.create_index(
        "ix_contact_requests_dedupe_hash_created_at",
        "contact_requests",
        ["dedupe_hash", "created_at"],
        unique=False,
    )


def downgrade():
    bind = op.get_bind()
    inspector = inspect(bind)
    foreign_keys = {fk["name"] for fk in inspector.get_foreign_keys("contact_requests")}
    if "fk_contact_requests_user_id_users" in foreign_keys:
        op.drop_constraint(
            "fk_contact_requests_user_id_users",
            "contact_requests",
            type_="foreignkey",
        )
    op.drop_index("ix_contact_requests_dedupe_hash_created_at", table_name="contact_requests")
    op.drop_index("ix_contact_requests_user_id", table_name="contact_requests")
    op.drop_index("ix_contact_requests_trace_id", table_name="contact_requests")
    op.drop_index("ix_contact_requests_email", table_name="contact_requests")
    op.drop_index("ix_contact_requests_category_status", table_name="contact_requests")
    op.drop_index("ix_contact_requests_created_at", table_name="contact_requests")
    op.drop_table("contact_requests")
