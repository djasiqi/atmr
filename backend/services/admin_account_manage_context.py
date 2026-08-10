"""Contexte drawer gestion compte admin."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from ext import db
from models.company import Company
from models.control_plane import (
    OrganizationMembership,
    OrganizationServiceEntitlement,
    PlatformOrganization,
    RoleTemplate,
    ServiceCatalog,
)
from models.driver import Driver
from models.enums import InstitutionRole
from models.institution import Institution
from models.refresh_token import RefreshToken
from models.user import User
from services.admin_account_role_transition import list_transport_tenants_for_picker
from services.admin_authz import (
    CAP_BILLING_LOCK,
    CAP_USERS_MANAGE,
    CAP_USERS_SECURITY,
    user_has_admin_capability,
)
from services.admin_partners_organizations import build_account_integrity
from services.admin_role_utils import normalized_role_value
from services.control_plane.classification import (
    CompanyProjectionKind,
    classify_company_for_control_plane,
)

_SERVICE_LABELS = {
    "company.own_portfolio": "Portefeuille propre",
    "company.marketplace": "Marketplace",
    "company.dispatch": "Dispatch",
    "company.driver_management": "Gestion des chauffeurs",
    "company.live_tracking": "Suivi en temps réel",
}


def _driver_counts(company_id: int) -> tuple[int, int]:
    total = (
        db.session.scalar(
            select(func.count())
            .select_from(Driver)
            .where(Driver.company_id == company_id)
        )
        or 0
    )
    active = (
        db.session.scalar(
            select(func.count())
            .select_from(Driver)
            .where(Driver.company_id == company_id, Driver.is_active.is_(True))
        )
        or 0
    )
    return int(active), int(total)


def _detected_services(company: Company) -> dict[str, Any]:
    org = db.session.scalar(
        select(PlatformOrganization).where(
            PlatformOrganization.company_id == company.id
        )
    )
    services: list[dict[str, Any]] = []
    if org is not None:
        rows = db.session.execute(
            select(
                ServiceCatalog.service_key,
                OrganizationServiceEntitlement.enforcement_mode,
            )
            .join(
                OrganizationServiceEntitlement,
                OrganizationServiceEntitlement.service_catalog_id == ServiceCatalog.id,
            )
            .where(OrganizationServiceEntitlement.organization_id == org.id)
        ).all()
        for key, mode in rows:
            if key in _SERVICE_LABELS:
                services.append(
                    {
                        "service_key": key,
                        "label": _SERVICE_LABELS[key],
                        "detected": True,
                        "enforcement_mode": mode or "shadow",
                    }
                )
    # Fallback dérivé legacy si rien projeté
    if not services:
        derived = [
            "company.own_portfolio",
            "company.driver_management",
            "company.live_tracking",
        ]
        if bool(getattr(company, "is_partner", False)) or bool(company.is_approved):
            derived.append("company.marketplace")
        if bool(company.dispatch_enabled):
            derived.append("company.dispatch")
        for key in derived:
            services.append(
                {
                    "service_key": key,
                    "label": _SERVICE_LABELS[key],
                    "detected": True,
                    "enforcement_mode": "shadow",
                }
            )
    return {
        "decision_mode": "shadow",
        "notice": (
            "Ces services sont détectés depuis la configuration legacy. "
            "Ils n'autorisent ni ne bloquent encore les fonctions de l'entreprise."
        ),
        "services": services,
    }


def _resolve_owned_transport_tenant(user_id: int) -> Company | None:
    owned = db.session.scalars(select(Company).where(Company.user_id == user_id)).all()
    transport = [
        c
        for c in owned
        if classify_company_for_control_plane(c).kind
        == CompanyProjectionKind.TRANSPORT_TENANT
    ]
    if len(transport) == 1:
        return transport[0]
    return None


def build_account_manage_context(
    user_id: int, *, actor_admin_id: int | None
) -> dict[str, Any] | None:
    user = db.session.get(User, user_id)
    if user is None:
        return None

    driver = db.session.scalar(select(Driver).where(Driver.user_id == user_id))
    integrity = build_account_integrity(user_id) or {}

    memberships = db.session.scalars(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.membership_status != "removed",
        )
    ).all()
    membership_payload = []
    for m in memberships:
        role = (
            db.session.get(RoleTemplate, m.role_template_id)
            if m.role_template_id
            else None
        )
        membership_payload.append(
            {
                "organization_id": m.organization_id,
                "role_key": role.role_key if role else None,
                "membership_status": m.membership_status,
                "source": m.source,
            }
        )

    company_id = None
    if driver and driver.company_id:
        company_id = int(driver.company_id)
    elif integrity.get("account", {}).get("company_id"):
        company_id = int(integrity["account"]["company_id"])

    role_s = normalized_role_value(user.role)
    company_profile = None
    commercial_restriction = None
    commercial_access = None  # compat FE legacy
    detected_services = None

    if role_s == "COMPANY":
        company = _resolve_owned_transport_tenant(user_id)
        if company is not None:
            active_drivers, total_drivers = _driver_counts(int(company.id))
            company_profile = {
                "company_id": company.id,
                "name": company.name,
                "contact_email": company.contact_email,
                "is_approved": bool(company.is_approved),
                "dispatch_enabled": bool(company.dispatch_enabled),
                "platform_suspended": bool(
                    getattr(company, "platform_suspended", False)
                ),
                "active_drivers_count": active_drivers,
                "total_drivers_count": total_drivers,
                "inactive_drivers_count": max(0, total_drivers - active_drivers),
            }
            state = getattr(company, "platform_billing_access_state", None) or "active"
            commercial_restriction = {
                "company_id": company.id,
                "state": state,
                "source": getattr(company, "platform_billing_state_source", None),
                "reason_code": getattr(
                    company, "platform_billing_state_reason_code", None
                ),
                "since": (
                    company.platform_billing_state_since.isoformat()
                    if getattr(company, "platform_billing_state_since", None)
                    else None
                ),
                "dunning_paused_until": (
                    company.dunning_paused_until.isoformat()
                    if getattr(company, "dunning_paused_until", None)
                    else None
                ),
                "dunning_pause_reason": getattr(company, "dunning_pause_reason", None),
            }
            # Compat anciens clients
            commercial_access = {
                "company_id": company.id,
                "company_name": company.name,
                "platform_billing_access_state": state,
                "dunning_paused_until": commercial_restriction["dunning_paused_until"],
            }
            detected_services = _detected_services(company)
            company_id = int(company.id)

    driver_profile = None
    if driver is not None:
        company_name = None
        if driver.company_id:
            co = db.session.get(Company, driver.company_id)
            company_name = co.name if co else None
        driver_type = getattr(driver, "driver_type", None)
        if hasattr(driver_type, "value"):
            driver_type = driver_type.value
        driver_profile = {
            "driver_id": int(driver.id),
            "company_id": int(driver.company_id) if driver.company_id else None,
            "company_name": company_name,
            "is_active": bool(driver.is_active),
            "is_available": bool(getattr(driver, "is_available", False)),
            "driver_type": str(driver_type) if driver_type else None,
        }

    sessions = (
        db.session.scalar(
            select(func.count())
            .select_from(RefreshToken)
            .where(RefreshToken.user_id == user_id)
        )
        or 0
    )

    institutions = db.session.execute(
        select(Institution.id, Institution.name).order_by(Institution.name)
    ).all()

    can_security = bool(
        actor_admin_id and user_has_admin_capability(actor_admin_id, CAP_USERS_SECURITY)
    )
    can_manage = bool(
        actor_admin_id and user_has_admin_capability(actor_admin_id, CAP_USERS_MANAGE)
    )
    can_billing = bool(
        company_profile
        and actor_admin_id
        and user_has_admin_capability(actor_admin_id, CAP_BILLING_LOCK)
    )

    return {
        "account": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "role": role_s,
            "account_status": user.account_status,
            "force_password_change": bool(
                getattr(user, "force_password_change", False)
            ),
            "created_at": user.created_at.isoformat() if user.created_at else None,
        },
        "legacy_context": {
            "company_id": company_id,
            "driver_id": int(driver.id) if driver else None,
            "institution_id": user.institution_id,
            "institution_role": user.institution_role,
        },
        "driver_profile": driver_profile,
        "company_profile": company_profile,
        "commercial_restriction": commercial_restriction,
        "detected_services": detected_services,
        "memberships": membership_payload,
        "commercial_access": commercial_access,
        "security": {
            "active_sessions": int(sessions),
            "password_temporary": bool(getattr(user, "force_password_change", False)),
        },
        "diagnostic": {
            "checks": integrity.get("checks") or [],
            "dependencies": integrity.get("dependencies") or {},
            "possible_matches": integrity.get("possible_matches") or [],
            "configuration_status": integrity.get("configuration_status"),
        },
        "allowed_actions": {
            "reset_password": can_security,
            "revoke_sessions": can_security,
            "change_role": can_manage,
            "change_driver_status": bool(
                can_manage and role_s == "DRIVER" and driver_profile is not None
            ),
            "manage_billing_access": can_billing,
            "manage_commercial_restriction": can_billing,
            "pause_dunning": can_billing,
            "manage_operational_flags": bool(
                can_manage and company_profile is not None
            ),
            "open_billing_configuration": bool(company_profile is not None),
            "open_platform_operations": bool(company_profile is not None),
        },
        "role_transition_options": {
            "transport_tenants": list_transport_tenants_for_picker(),
            "institutions": [{"id": i, "name": n} for i, n in institutions],
            "institution_roles": InstitutionRole.choices(),
        },
    }
