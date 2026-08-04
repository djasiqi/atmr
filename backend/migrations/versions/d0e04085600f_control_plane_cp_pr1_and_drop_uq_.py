"""control_plane_cp_pr1

Revision ID: d0e04085600f
Revises: e4f273565844
Create Date: 2026-08-03 22:01:05.839030

CP-PR1 : tables control plane + colonnes data_origin User.
Ne crée / ne retire aucune contrainte UNIQUE sur company.user_id
(coquilles cliniques partagent volontairement le propriétaire).
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "d0e04085600f"
down_revision = "e4f273565844"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")

    op.create_table(
        "permission_catalog",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("permission_key", sa.String(length=128), nullable=False),
        sa.Column("required_service_key", sa.String(length=128), nullable=True),
        sa.Column("action_type", sa.String(length=32), nullable=True),
        sa.Column("sensitivity", sa.String(length=32), nullable=True),
        sa.Column(
            "policy_verification",
            sa.String(length=32),
            server_default="frontend_only",
            nullable=False,
        ),
        sa.Column("label", sa.String(length=200), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "permission_key", name="uq_permission_catalog_permission_key"
        ),
    )
    op.create_table(
        "role_template",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("organization_type", sa.String(length=32), nullable=False),
        sa.Column("role_key", sa.String(length=64), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_type",
            "role_key",
            name="uq_role_template_org_type_role_key",
        ),
    )
    op.create_table(
        "service_catalog",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("service_key", sa.String(length=128), nullable=False),
        sa.Column("organization_type", sa.String(length=32), nullable=False),
        sa.Column("label", sa.String(length=200), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "dependencies_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "is_sensitive", sa.Boolean(), server_default="false", nullable=False
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("service_key", name="uq_service_catalog_service_key"),
    )
    op.create_table(
        "role_template_permission",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("role_template_id", sa.Integer(), nullable=False),
        sa.Column("permission_catalog_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["permission_catalog_id"],
            ["permission_catalog.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["role_template_id"], ["role_template.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "role_template_id",
            "permission_catalog_id",
            name="uq_role_template_permission",
        ),
    )
    op.create_index(
        "ix_role_template_permission_permission_catalog_id",
        "role_template_permission",
        ["permission_catalog_id"],
    )
    op.create_index(
        "ix_role_template_permission_role_template_id",
        "role_template_permission",
        ["role_template_id"],
    )

    op.create_table(
        "control_plane_entity_override",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("entity_id", sa.Integer(), nullable=False),
        sa.Column("override_key", sa.String(length=64), nullable=False),
        sa.Column("override_value", sa.String(length=64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "entity_type",
            "entity_id",
            "override_key",
            name="uq_control_plane_entity_override",
        ),
    )

    op.create_table(
        "platform_organization",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column(
            "public_id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("organization_type", sa.String(length=32), nullable=False),
        sa.Column("company_id", sa.Integer(), nullable=True),
        sa.Column("institution_id", sa.Integer(), nullable=True),
        sa.Column(
            "lifecycle_status",
            sa.String(length=32),
            server_default="onboarding",
            nullable=False,
        ),
        sa.Column(
            "lifecycle_source",
            sa.String(length=32),
            server_default="legacy_derived",
            nullable=False,
        ),
        sa.Column(
            "data_origin",
            sa.String(length=32),
            server_default="unknown",
            nullable=False,
        ),
        sa.Column("data_origin_source", sa.String(length=64), nullable=True),
        sa.Column("data_origin_confidence", sa.String(length=32), nullable=True),
        sa.Column("classified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("classified_by_user_id", sa.Integer(), nullable=True),
        sa.Column(
            "classification_evidence_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("archived_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint(
            "("
            "(company_id IS NOT NULL AND institution_id IS NULL "
            "AND organization_type = 'company') OR "
            "(institution_id IS NOT NULL AND company_id IS NULL "
            "AND organization_type = 'institution')"
            ")",
            name="ck_platform_organization_xor_type",
        ),
        sa.ForeignKeyConstraint(
            ["classified_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["company_id"], ["company.id"], ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["institution_id"], ["institutions.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "company_id", name="uq_platform_organization_company_id"
        ),
        sa.UniqueConstraint(
            "institution_id", name="uq_platform_organization_institution_id"
        ),
        sa.UniqueConstraint(
            "public_id", name="uq_platform_organization_public_id"
        ),
    )
    op.create_index(
        "ix_platform_organization_company_id",
        "platform_organization",
        ["company_id"],
    )
    op.create_index(
        "ix_platform_organization_institution_id",
        "platform_organization",
        ["institution_id"],
    )

    op.create_table(
        "control_plane_anomaly",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("fingerprint", sa.String(length=128), nullable=False),
        sa.Column("code", sa.String(length=96), nullable=False),
        sa.Column("severity", sa.String(length=16), nullable=False),
        sa.Column("entity_type", sa.String(length=32), nullable=False),
        sa.Column("entity_key", sa.String(length=128), nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column(
            "details_json",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "first_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "last_seen_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_source", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["platform_organization.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "fingerprint", name="uq_control_plane_anomaly_fingerprint"
        ),
    )
    op.create_index(
        "ix_control_plane_anomaly_code", "control_plane_anomaly", ["code"]
    )
    op.create_index(
        "ix_control_plane_anomaly_organization_id",
        "control_plane_anomaly",
        ["organization_id"],
    )
    op.create_index(
        "ix_control_plane_anomaly_user_id",
        "control_plane_anomaly",
        ["user_id"],
    )

    op.create_table(
        "organization_membership",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("role_template_id", sa.Integer(), nullable=True),
        sa.Column(
            "membership_status",
            sa.String(length=32),
            server_default="active",
            nullable=False,
        ),
        sa.Column(
            "scope_type",
            sa.String(length=64),
            server_default="organization",
            nullable=False,
        ),
        sa.Column(
            "scope_schema_version",
            sa.Integer(),
            server_default="1",
            nullable=False,
        ),
        sa.Column(
            "scope_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("source", sa.String(length=64), nullable=True),
        sa.Column("invited_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("suspended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("removed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["platform_organization.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["role_template_id"], ["role_template.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["user_id"], ["user.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_id",
            "user_id",
            name="uq_organization_membership_org_user",
        ),
    )
    op.create_index(
        "ix_organization_membership_organization_id",
        "organization_membership",
        ["organization_id"],
    )
    op.create_index(
        "ix_organization_membership_role_template_id",
        "organization_membership",
        ["role_template_id"],
    )
    op.create_index(
        "ix_organization_membership_user_id",
        "organization_membership",
        ["user_id"],
    )

    op.create_table(
        "organization_service_entitlement",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("organization_id", sa.Integer(), nullable=False),
        sa.Column("service_catalog_id", sa.Integer(), nullable=False),
        sa.Column(
            "status", sa.String(length=32), server_default="enabled", nullable=False
        ),
        sa.Column("starts_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ends_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "source",
            sa.String(length=64),
            server_default="legacy_inferred",
            nullable=False,
        ),
        sa.Column(
            "enforcement_mode",
            sa.String(length=32),
            server_default="shadow",
            nullable=False,
        ),
        sa.Column(
            "confidence",
            sa.String(length=32),
            server_default="heuristic",
            nullable=False,
        ),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("configured_by_user_id", sa.Integer(), nullable=True),
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
        sa.ForeignKeyConstraint(
            ["configured_by_user_id"], ["user.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["organization_id"],
            ["platform_organization.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["service_catalog_id"],
            ["service_catalog.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "organization_id",
            "service_catalog_id",
            name="uq_org_service_entitlement_org_service",
        ),
    )
    op.create_index(
        "ix_organization_service_entitlement_organization_id",
        "organization_service_entitlement",
        ["organization_id"],
    )
    op.create_index(
        "ix_organization_service_entitlement_service_catalog_id",
        "organization_service_entitlement",
        ["service_catalog_id"],
    )

    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "data_origin",
                sa.String(length=32),
                server_default="unknown",
                nullable=False,
            )
        )
        batch_op.add_column(
            sa.Column("data_origin_source", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "data_origin_confidence", sa.String(length=32), nullable=True
            )
        )
        batch_op.add_column(
            sa.Column("classified_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("classified_by_user_id", sa.Integer(), nullable=True)
        )
        batch_op.add_column(
            sa.Column(
                "classification_evidence_json",
                postgresql.JSONB(astext_type=sa.Text()),
                nullable=True,
            )
        )
        batch_op.create_index("idx_user_data_origin", ["data_origin"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("user", schema=None) as batch_op:
        batch_op.drop_index("idx_user_data_origin")
        batch_op.drop_column("classification_evidence_json")
        batch_op.drop_column("classified_by_user_id")
        batch_op.drop_column("classified_at")
        batch_op.drop_column("data_origin_confidence")
        batch_op.drop_column("data_origin_source")
        batch_op.drop_column("data_origin")

    op.drop_index(
        "ix_organization_service_entitlement_service_catalog_id",
        table_name="organization_service_entitlement",
    )
    op.drop_index(
        "ix_organization_service_entitlement_organization_id",
        table_name="organization_service_entitlement",
    )
    op.drop_table("organization_service_entitlement")

    op.drop_index(
        "ix_organization_membership_user_id", table_name="organization_membership"
    )
    op.drop_index(
        "ix_organization_membership_role_template_id",
        table_name="organization_membership",
    )
    op.drop_index(
        "ix_organization_membership_organization_id",
        table_name="organization_membership",
    )
    op.drop_table("organization_membership")

    op.drop_index(
        "ix_control_plane_anomaly_user_id", table_name="control_plane_anomaly"
    )
    op.drop_index(
        "ix_control_plane_anomaly_organization_id",
        table_name="control_plane_anomaly",
    )
    op.drop_index(
        "ix_control_plane_anomaly_code", table_name="control_plane_anomaly"
    )
    op.drop_table("control_plane_anomaly")

    op.drop_index(
        "ix_platform_organization_institution_id",
        table_name="platform_organization",
    )
    op.drop_index(
        "ix_platform_organization_company_id", table_name="platform_organization"
    )
    op.drop_table("platform_organization")
    op.drop_table("control_plane_entity_override")

    op.drop_index(
        "ix_role_template_permission_role_template_id",
        table_name="role_template_permission",
    )
    op.drop_index(
        "ix_role_template_permission_permission_catalog_id",
        table_name="role_template_permission",
    )
    op.drop_table("role_template_permission")
    op.drop_table("service_catalog")
    op.drop_table("role_template")
    op.drop_table("permission_catalog")
