"""Tables platform_runbook_execution et platform_change_request (gouvernance V1).

Revision ID: 20260329_plat_gov_tbls
Revises: 20260328_co_plat_susp
Create Date: 2026-03-29

"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "20260329_plat_gov_tbls"
down_revision = "20260328_co_plat_susp"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "platform_runbook_execution",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("runbook_id", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=True),
        sa.Column("correlation_id", sa.String(length=128), nullable=True),
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
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("triggered_by_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "preview_snapshot_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("rollback_of_execution_id", sa.String(length=36), nullable=True),
        sa.ForeignKeyConstraint(
            ["rollback_of_execution_id"],
            ["platform_runbook_execution.id"],
            name=op.f("fk_prunbook_exec_rollback_of"),
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["company.id"],
            name=op.f("fk_prunbook_exec_tenant"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["triggered_by_user_id"],
            ["user.id"],
            name=op.f("fk_prunbook_exec_triggered_by"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_platform_runbook_execution")),
    )
    op.create_index(
        "ix_prunbook_exec_correlation",
        "platform_runbook_execution",
        ["correlation_id"],
        unique=False,
    )
    op.create_index(
        "ix_prunbook_exec_runbook_status",
        "platform_runbook_execution",
        ["runbook_id", "status"],
        unique=False,
    )
    op.create_index(
        "ix_prunbook_exec_tenant_created",
        "platform_runbook_execution",
        ["tenant_id", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_platform_runbook_execution_status"),
        "platform_runbook_execution",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_platform_runbook_execution_tenant_id"),
        "platform_runbook_execution",
        ["tenant_id"],
        unique=False,
    )

    op.create_table(
        "platform_change_request",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("change_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("tenant_id", sa.Integer(), nullable=True),
        sa.Column("correlation_id", sa.String(length=128), nullable=True),
        sa.Column("requested_by_user_id", sa.Integer(), nullable=True),
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
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("effective_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("justification", sa.Text(), server_default="", nullable=False),
        sa.Column("incident_id", sa.String(length=128), nullable=True),
        sa.Column(
            "target_snapshot_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "result_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "metadata_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.ForeignKeyConstraint(
            ["requested_by_user_id"],
            ["user.id"],
            name=op.f("fk_pchreq_requested_by"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["company.id"],
            name=op.f("fk_pchreq_tenant"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_platform_change_request")),
    )
    op.create_index(
        "ix_pchreq_change_type",
        "platform_change_request",
        ["change_type"],
        unique=False,
    )
    op.create_index(
        "ix_pchreq_correlation",
        "platform_change_request",
        ["correlation_id"],
        unique=False,
    )
    op.create_index(
        "ix_pchreq_tenant_created",
        "platform_change_request",
        ["tenant_id", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_platform_change_request_status"),
        "platform_change_request",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_platform_change_request_tenant_id"),
        "platform_change_request",
        ["tenant_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_platform_change_request_tenant_id"),
        table_name="platform_change_request",
    )
    op.drop_index(
        op.f("ix_platform_change_request_status"), table_name="platform_change_request"
    )
    op.drop_index("ix_pchreq_tenant_created", table_name="platform_change_request")
    op.drop_index("ix_pchreq_correlation", table_name="platform_change_request")
    op.drop_index("ix_pchreq_change_type", table_name="platform_change_request")
    op.drop_table("platform_change_request")

    op.drop_index(
        op.f("ix_platform_runbook_execution_tenant_id"),
        table_name="platform_runbook_execution",
    )
    op.drop_index(
        op.f("ix_platform_runbook_execution_status"),
        table_name="platform_runbook_execution",
    )
    op.drop_index(
        "ix_prunbook_exec_tenant_created", table_name="platform_runbook_execution"
    )
    op.drop_index(
        "ix_prunbook_exec_runbook_status", table_name="platform_runbook_execution"
    )
    op.drop_index(
        "ix_prunbook_exec_correlation", table_name="platform_runbook_execution"
    )
    op.drop_table("platform_runbook_execution")
