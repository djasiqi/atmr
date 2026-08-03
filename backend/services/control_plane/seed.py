"""Seed idempotent des catalogues control plane (CP-PR1)."""

from __future__ import annotations

import logging
from typing import Any

from ext import db
from models.control_plane import (
    PermissionCatalog,
    RoleTemplate,
    RoleTemplatePermission,
    ServiceCatalog,
)

logger = logging.getLogger(__name__)

# Prestations (clés distinctes des permissions)
SERVICE_SEED: list[dict[str, Any]] = [
    {
        "service_key": "institution.transport_coordination",
        "organization_type": "institution",
        "label": "Coordination des transports",
        "description": "Demandes, suivi des statuts, messagerie",
    },
    {
        "service_key": "institution.patient_management",
        "organization_type": "institution",
        "label": "Patients et bénéficiaires",
    },
    {
        "service_key": "institution.live_tracking",
        "organization_type": "institution",
        "label": "Suivi en temps réel",
    },
    {
        "service_key": "institution.billing",
        "organization_type": "institution",
        "label": "Facturation institutionnelle",
        "is_sensitive": True,
    },
    {
        "service_key": "institution.exports",
        "organization_type": "institution",
        "label": "Rapports et exports",
        "is_sensitive": True,
    },
    {
        "service_key": "institution.api",
        "organization_type": "institution",
        "label": "Intégrations API",
        "is_sensitive": True,
    },
    {
        "service_key": "institution.users_teams",
        "organization_type": "institution",
        "label": "Utilisateurs et équipes",
    },
    {
        "service_key": "company.own_portfolio",
        "organization_type": "company",
        "label": "Portefeuille propre",
    },
    {
        "service_key": "company.marketplace",
        "organization_type": "company",
        "label": "Réseau LIRIE / Marketplace",
    },
    {
        "service_key": "company.dispatch",
        "organization_type": "company",
        "label": "Dispatch et exploitation",
    },
    {
        "service_key": "company.driver_management",
        "organization_type": "company",
        "label": "Gestion des chauffeurs",
    },
    {
        "service_key": "company.live_tracking",
        "organization_type": "company",
        "label": "Géolocalisation",
    },
    {
        "service_key": "company.billing",
        "organization_type": "company",
        "label": "Facturation",
        "is_sensitive": True,
    },
    {
        "service_key": "company.pricing",
        "organization_type": "company",
        "label": "Tarification",
    },
    {
        "service_key": "company.analytics",
        "organization_type": "company",
        "label": "Analytics",
    },
    {
        "service_key": "company.api",
        "organization_type": "company",
        "label": "Intégrations API",
        "is_sensitive": True,
    },
]

ROLE_SEED: list[dict[str, str]] = [
    {
        "organization_type": "institution",
        "role_key": "institution_admin",
        "label": "Administrateur institution",
    },
    {
        "organization_type": "institution",
        "role_key": "institution_requester",
        "label": "Demandeur",
    },
    {
        "organization_type": "institution",
        "role_key": "institution_reader",
        "label": "Lecteur",
    },
    {
        "organization_type": "institution",
        "role_key": "institution_billing",
        "label": "Facturation",
    },
    {
        "organization_type": "institution",
        "role_key": "institution_curator",
        "label": "Curateur",
    },
    {
        "organization_type": "institution",
        "role_key": "institution_reception",
        "label": "Réception",
    },
    {
        "organization_type": "institution",
        "role_key": "legacy_unresolved",
        "label": "Rôle non résolu (legacy)",
    },
    {
        "organization_type": "company",
        "role_key": "company_owner",
        "label": "Propriétaire / Administrateur",
    },
    {
        "organization_type": "company",
        "role_key": "company_driver",
        "label": "Chauffeur",
    },
    {
        "organization_type": "company",
        "role_key": "company_dispatcher",
        "label": "Dispatcher",
    },
    {
        "organization_type": "company",
        "role_key": "company_billing",
        "label": "Facturation entreprise",
    },
    {
        "organization_type": "company",
        "role_key": "company_reader",
        "label": "Lecteur / Analyste",
    },
]

# Aligné sur institutionPermissions.js + contrôles backend connus
# permission_key, required_service_key, policy_verification
PERMISSION_SEED: list[dict[str, Any]] = [
    {
        "permission_key": "institution.requests.view",
        "required_service_key": "institution.transport_coordination",
        "policy_verification": "backend_verified",
    },
    {
        "permission_key": "institution.requests.create",
        "required_service_key": "institution.transport_coordination",
        "policy_verification": "backend_verified",
    },
    {
        "permission_key": "institution.requests.edit",
        "required_service_key": "institution.transport_coordination",
        "policy_verification": "backend_verified",
    },
    {
        "permission_key": "institution.requests.send",
        "required_service_key": "institution.transport_coordination",
        "policy_verification": "backend_verified",
    },
    {
        "permission_key": "institution.requests.cancel",
        "required_service_key": "institution.transport_coordination",
        "policy_verification": "backend_verified",
    },
    {
        "permission_key": "institution.patients.view",
        "required_service_key": "institution.patient_management",
        "policy_verification": "backend_verified",
    },
    {
        "permission_key": "institution.patients.create",
        "required_service_key": "institution.patient_management",
        "policy_verification": "backend_verified",
    },
    {
        "permission_key": "institution.patients.edit",
        "required_service_key": "institution.patient_management",
        "policy_verification": "backend_verified",
    },
    {
        "permission_key": "institution.settings.view",
        "required_service_key": None,
        "policy_verification": "frontend_only",
    },
    {
        "permission_key": "institution.settings.edit_preferences",
        "required_service_key": None,
        "policy_verification": "backend_verified",
    },
    {
        "permission_key": "institution.api.manage_keys",
        "required_service_key": "institution.api",
        "policy_verification": "backend_verified",
        "sensitivity": "high",
    },
    {
        "permission_key": "institution.billing.edit",
        "required_service_key": "institution.billing",
        "policy_verification": "backend_verified",
        "sensitivity": "high",
    },
    {
        "permission_key": "institution.billing.edit_request",
        "required_service_key": "institution.billing",
        "policy_verification": "backend_verified",
        "sensitivity": "high",
    },
    {
        "permission_key": "institution.admin_data.view",
        "required_service_key": "institution.patient_management",
        "policy_verification": "backend_verified",
        "sensitivity": "high",
    },
    {
        "permission_key": "institution.admin_data.edit",
        "required_service_key": "institution.patient_management",
        "policy_verification": "backend_verified",
        "sensitivity": "high",
    },
    {
        "permission_key": "institution.patients.edit_billing_data",
        "required_service_key": "institution.billing",
        "policy_verification": "frontend_only",
        "sensitivity": "high",
    },
    {
        "permission_key": "institution.exports.transports",
        "required_service_key": "institution.exports",
        "policy_verification": "backend_verified",
        "sensitivity": "high",
    },
    {
        "permission_key": "company.bookings.view",
        "required_service_key": "company.own_portfolio",
        "policy_verification": "backend_verified",
    },
    {
        "permission_key": "company.dispatch.assign",
        "required_service_key": "company.dispatch",
        "policy_verification": "backend_verified",
    },
    {
        "permission_key": "company.drivers.manage",
        "required_service_key": "company.driver_management",
        "policy_verification": "backend_verified",
    },
    {
        "permission_key": "company.billing.export",
        "required_service_key": "company.billing",
        "policy_verification": "frontend_only",
        "sensitivity": "high",
    },
]

# Mapping rôle institution → permissions (miroir frontend)
ROLE_PERMISSION_KEYS: dict[str, list[str]] = {
    "institution_admin": [
        "institution.requests.view",
        "institution.requests.create",
        "institution.requests.edit",
        "institution.requests.send",
        "institution.requests.cancel",
        "institution.patients.view",
        "institution.patients.create",
        "institution.patients.edit",
        "institution.settings.view",
        "institution.settings.edit_preferences",
        "institution.api.manage_keys",
        "institution.billing.edit",
        "institution.billing.edit_request",
        "institution.admin_data.view",
        "institution.admin_data.edit",
        "institution.patients.edit_billing_data",
        "institution.exports.transports",
    ],
    "institution_requester": [
        "institution.requests.view",
        "institution.requests.create",
        "institution.requests.edit",
        "institution.requests.send",
        "institution.requests.cancel",
        "institution.patients.view",
        "institution.patients.create",
        "institution.patients.edit",
        "institution.settings.view",
    ],
    "institution_reader": [
        "institution.requests.view",
        "institution.patients.view",
        "institution.settings.view",
    ],
    "institution_billing": [
        "institution.requests.view",
        "institution.requests.create",
        "institution.requests.edit",
        "institution.requests.send",
        "institution.requests.cancel",
        "institution.patients.view",
        "institution.patients.create",
        "institution.patients.edit",
        "institution.settings.view",
        "institution.billing.edit",
        "institution.billing.edit_request",
        "institution.admin_data.view",
        "institution.admin_data.edit",
        "institution.patients.edit_billing_data",
        "institution.exports.transports",
    ],
    "institution_reception": [
        "institution.requests.view",
        "institution.patients.view",
        "institution.settings.view",
        "institution.exports.transports",
    ],
    "institution_curator": [
        "institution.requests.view",
        "institution.requests.create",
        "institution.requests.edit",
        "institution.requests.send",
        "institution.requests.cancel",
        "institution.patients.view",
        "institution.patients.create",
        "institution.patients.edit",
        "institution.billing.edit",
        "institution.billing.edit_request",
        "institution.settings.view",
        "institution.admin_data.view",
        "institution.admin_data.edit",
        "institution.patients.edit_billing_data",
    ],
    "company_owner": [
        "company.bookings.view",
        "company.dispatch.assign",
        "company.drivers.manage",
        "company.billing.export",
    ],
    "company_driver": [
        "company.bookings.view",
    ],
}


def seed_control_plane_catalogs(*, commit: bool = True) -> dict[str, int]:
    """Insère ou met à jour catalogues / templates / permissions (idempotent)."""
    services_upserted = 0
    for row in SERVICE_SEED:
        existing = ServiceCatalog.query.filter_by(service_key=row["service_key"]).first()
        if existing is None:
            existing = ServiceCatalog(service_key=row["service_key"])
            db.session.add(existing)
        existing.organization_type = row["organization_type"]
        existing.label = row["label"]
        existing.description = row.get("description")
        existing.is_sensitive = bool(row.get("is_sensitive", False))
        services_upserted += 1

    roles_upserted = 0
    for row in ROLE_SEED:
        existing = RoleTemplate.query.filter_by(
            organization_type=row["organization_type"],
            role_key=row["role_key"],
        ).first()
        if existing is None:
            existing = RoleTemplate(
                organization_type=row["organization_type"],
                role_key=row["role_key"],
            )
            db.session.add(existing)
        existing.label = row["label"]
        roles_upserted += 1

    perms_upserted = 0
    for row in PERMISSION_SEED:
        existing = PermissionCatalog.query.filter_by(
            permission_key=row["permission_key"]
        ).first()
        if existing is None:
            existing = PermissionCatalog(permission_key=row["permission_key"])
            db.session.add(existing)
        existing.required_service_key = row.get("required_service_key")
        existing.policy_verification = row.get("policy_verification", "frontend_only")
        existing.sensitivity = row.get("sensitivity")
        existing.label = row["permission_key"]
        perms_upserted += 1

    db.session.flush()

    mappings = 0
    for role_key, perm_keys in ROLE_PERMISSION_KEYS.items():
        role = RoleTemplate.query.filter_by(role_key=role_key).first()
        if role is None:
            continue
        for pk in perm_keys:
            perm = PermissionCatalog.query.filter_by(permission_key=pk).first()
            if perm is None:
                continue
            link = RoleTemplatePermission.query.filter_by(
                role_template_id=role.id,
                permission_catalog_id=perm.id,
            ).first()
            if link is None:
                db.session.add(
                    RoleTemplatePermission(
                        role_template_id=role.id,
                        permission_catalog_id=perm.id,
                    )
                )
                mappings += 1

    if commit:
        db.session.commit()
    logger.info(
        "[control_plane.seed] services=%s roles=%s perms=%s new_mappings=%s",
        services_upserted,
        roles_upserted,
        perms_upserted,
        mappings,
    )
    return {
        "services": services_upserted,
        "roles": roles_upserted,
        "permissions": perms_upserted,
        "new_mappings": mappings,
    }
