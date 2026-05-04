"""add demo requests table

Revision ID: 20260302_demo_requests
Revises: 20260302_contact_requests
Create Date: 2026-03-02 19:10:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260302_demo_requests"
down_revision = "20260302_contact_requests"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "demo_requests",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("email", sa.String(length=254), nullable=False),
        sa.Column("phone", sa.String(length=32), nullable=True),
        sa.Column("organization", sa.String(length=180), nullable=False),
        sa.Column("organization_type", sa.String(length=64), nullable=False),
        sa.Column("use_case", sa.String(length=80), nullable=False),
        sa.Column("volume_range", sa.String(length=32), nullable=True),
        sa.Column("integration_required", sa.String(length=16), nullable=False),
        sa.Column("integration_system", sa.String(length=180), nullable=True),
        sa.Column("timing", sa.String(length=32), nullable=False),
        sa.Column("preferred_slot", sa.String(length=32), nullable=False),
        sa.Column("preferred_period", sa.String(length=16), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("score", sa.Integer(), server_default="0", nullable=False),
        sa.Column("status", sa.String(length=32), server_default="new", nullable=False),
        sa.Column("trace_id", sa.String(length=64), nullable=False),
        sa.Column(
            "source",
            sa.String(length=64),
            server_default="web_demo_request",
            nullable=False,
        ),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=512), nullable=True),
        sa.Column("assigned_channel", sa.String(length=120), nullable=True),
        sa.Column(
            "email_delivery_status",
            sa.String(length=32),
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_demo_requests_created_at", "demo_requests", ["created_at"], unique=False
    )
    op.create_index(
        "ix_demo_requests_status", "demo_requests", ["status"], unique=False
    )
    op.create_index("ix_demo_requests_score", "demo_requests", ["score"], unique=False)
    op.create_index("ix_demo_requests_email", "demo_requests", ["email"], unique=False)
    op.create_index(
        "ix_demo_requests_trace_id", "demo_requests", ["trace_id"], unique=False
    )


def downgrade():
    op.drop_index("ix_demo_requests_trace_id", table_name="demo_requests")
    op.drop_index("ix_demo_requests_email", table_name="demo_requests")
    op.drop_index("ix_demo_requests_score", table_name="demo_requests")
    op.drop_index("ix_demo_requests_status", table_name="demo_requests")
    op.drop_index("ix_demo_requests_created_at", table_name="demo_requests")
    op.drop_table("demo_requests")
