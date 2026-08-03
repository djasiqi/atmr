"""Control plane partenaires / identités — CP-PR1 (projection diagnostique).

Legacy reste l'autorité des portails métier. Ces tables projettent organisations,
appartenances, catalogues et anomalies pour l'admin plateforme.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ext import db


class ServiceCatalog(db.Model):
    """Catalogue des prestations LIRIE (clés stables, distinctes des permissions)."""

    __tablename__ = "service_catalog"
    __table_args__ = (
        UniqueConstraint("service_key", name="uq_service_catalog_service_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    service_key: Mapped[str] = mapped_column(String(128), nullable=False)
    organization_type: Mapped[str] = mapped_column(
        String(32), nullable=False
    )  # company | institution
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    dependencies_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    is_sensitive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RoleTemplate(db.Model):
    """Templates de rôle par type d'organisation."""

    __tablename__ = "role_template"
    __table_args__ = (
        UniqueConstraint(
            "organization_type",
            "role_key",
            name="uq_role_template_org_type_role_key",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    organization_type: Mapped[str] = mapped_column(String(32), nullable=False)
    role_key: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class PermissionCatalog(db.Model):
    """Permissions granulaires ; required_service_key lie à une prestation."""

    __tablename__ = "permission_catalog"
    __table_args__ = (
        UniqueConstraint("permission_key", name="uq_permission_catalog_permission_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    permission_key: Mapped[str] = mapped_column(String(128), nullable=False)
    required_service_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    action_type: Mapped[str | None] = mapped_column(String(32), nullable=True)
    sensitivity: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # backend_verified | frontend_only | mismatch
    policy_verification: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="frontend_only"
    )
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RoleTemplatePermission(db.Model):
    """Mapping rôle template → permission."""

    __tablename__ = "role_template_permission"
    __table_args__ = (
        UniqueConstraint(
            "role_template_id",
            "permission_catalog_id",
            name="uq_role_template_permission",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    role_template_id: Mapped[int] = mapped_column(
        ForeignKey("role_template.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    permission_catalog_id: Mapped[int] = mapped_column(
        ForeignKey("permission_catalog.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )


class PlatformOrganization(db.Model):
    """Façade commune Company / Institution pour le control plane."""

    __tablename__ = "platform_organization"
    __table_args__ = (
        CheckConstraint(
            "("
            "(company_id IS NOT NULL AND institution_id IS NULL "
            "AND organization_type = 'company') OR "
            "(institution_id IS NOT NULL AND company_id IS NULL "
            "AND organization_type = 'institution')"
            ")",
            name="ck_platform_organization_xor_type",
        ),
        UniqueConstraint("public_id", name="uq_platform_organization_public_id"),
        UniqueConstraint("company_id", name="uq_platform_organization_company_id"),
        UniqueConstraint(
            "institution_id", name="uq_platform_organization_institution_id"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    public_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
        server_default=text("gen_random_uuid()"),
    )
    organization_type: Mapped[str] = mapped_column(String(32), nullable=False)
    company_id: Mapped[int | None] = mapped_column(
        ForeignKey("company.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    institution_id: Mapped[int | None] = mapped_column(
        ForeignKey("institutions.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    # draft | onboarding | active | suspended | archived
    lifecycle_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="onboarding"
    )
    # legacy_derived | explicit_admin
    lifecycle_source: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="legacy_derived"
    )
    # production | demo | test | internal | synthetic | unknown
    data_origin: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="unknown"
    )
    data_origin_source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    data_origin_confidence: Mapped[str | None] = mapped_column(String(32), nullable=True)
    classified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    classified_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    classification_evidence_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True
    )
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    suspended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    archived_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    memberships = relationship(
        "OrganizationMembership",
        back_populates="organization",
        cascade="all, delete-orphan",
    )
    entitlements = relationship(
        "OrganizationServiceEntitlement",
        back_populates="organization",
        cascade="all, delete-orphan",
    )


class OrganizationMembership(db.Model):
    """Appartenance User ↔ platform_organization."""

    __tablename__ = "organization_membership"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "user_id",
            name="uq_organization_membership_org_user",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("platform_organization.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id: Mapped[int] = mapped_column(
        ForeignKey("user.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    role_template_id: Mapped[int | None] = mapped_column(
        ForeignKey("role_template.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # invited | active | suspended | removed | needs_review
    membership_status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="active"
    )
    scope_type: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default="organization"
    )
    scope_schema_version: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default="1"
    )
    scope_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    # legacy_sync | invitation | explicit_admin | backfill
    source: Mapped[str | None] = mapped_column(String(64), nullable=True)
    invited_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    activated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    suspended_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    removed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    organization = relationship("PlatformOrganization", back_populates="memberships")
    role_template = relationship("RoleTemplate")


class OrganizationServiceEntitlement(db.Model):
    """Prestation activée (ou détectée en shadow) pour une organisation."""

    __tablename__ = "organization_service_entitlement"
    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "service_catalog_id",
            name="uq_org_service_entitlement_org_service",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    organization_id: Mapped[int] = mapped_column(
        ForeignKey("platform_organization.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    service_catalog_id: Mapped[int] = mapped_column(
        ForeignKey("service_catalog.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    # trial | enabled | suspended | expired | disabled
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="enabled"
    )
    starts_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    ends_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # explicit_admin | contract | legacy_observed | legacy_inferred | demo_provisioning
    source: Mapped[str] = mapped_column(
        String(64), nullable=False, server_default="legacy_inferred"
    )
    # shadow | enforced — CP-PR1 : toujours shadow pour le backfill
    enforcement_mode: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="shadow"
    )
    # explicit | derived | heuristic
    confidence: Mapped[str] = mapped_column(
        String(32), nullable=False, server_default="heuristic"
    )
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    configured_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    organization = relationship("PlatformOrganization", back_populates="entitlements")
    service = relationship("ServiceCatalog")


class ControlPlaneAnomaly(db.Model):
    """File d'anomalies persistées (reconcile)."""

    __tablename__ = "control_plane_anomaly"
    __table_args__ = (
        UniqueConstraint("fingerprint", name="uq_control_plane_anomaly_fingerprint"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    code: Mapped[str] = mapped_column(String(96), nullable=False, index=True)
    # critical | warning | info
    severity: Mapped[str] = mapped_column(String(16), nullable=False)
    # account | organization | membership | projection | permission
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_key: Mapped[str] = mapped_column(String(128), nullable=False)
    organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("platform_organization.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    details_json: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    resolved_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    resolution_source: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ControlPlaneEntityOverride(db.Model):
    """Overrides admin pour classification fail-closed (Company kind, etc.)."""

    __tablename__ = "control_plane_entity_override"
    __table_args__ = (
        UniqueConstraint(
            "entity_type",
            "entity_id",
            "override_key",
            name="uq_control_plane_entity_override",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[int] = mapped_column(Integer, nullable=False)
    override_key: Mapped[str] = mapped_column(
        String(64), nullable=False
    )  # company_projection_kind | data_origin
    override_value: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_user_id: Mapped[int | None] = mapped_column(
        ForeignKey("user.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
