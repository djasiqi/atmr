"""Périmètre d'accès validé, un portail à la fois.

Le drapeau ``PORTAL_GRANT_ENFORCEMENT`` reste éteint par défaut. Dans ce cas,
la réponse reste le comportement historique légitime : rôle, société unique,
chauffeur ou institution déjà liés. Un droit ``portal_grant`` actif n'ouvre
rien de plus.

Deux protections s'appliquent même drapeau éteint :

- une ligne ``revoked`` interdit ce portail et cette organisation ;
- un ``company_id`` présenté (JWT ou en-tête) qui n'est pas l'organisation
  autorisée est refusé. On ne prend jamais la première société trouvée.

Drapeau allumé, dans les tests : seul un droit ``active`` ouvre son périmètre.
``pending_activation`` n'ouvre pas ce portail et ne retire pas les autres.
Un compte sans ligne n'hérite du rôle que s'il est explicitement éligible
à la transition pour ce portail (``PORTAL_GRANT_LEGACY_CUTOFF``).

``PORTAL_GRANT_OBSERVATION`` journalise les divergences sans changer la réponse.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from flask import has_request_context, request
from sqlalchemy.exc import ProgrammingError

from ext import db
from models import Company, Driver, Institution, User, UserRole
from models.enums import PortalGrantStatus, PortalKind
from models.portal_grant import PortalGrant

logger = logging.getLogger(__name__)

_ROLE_TO_PORTAL = {
    "CLIENT": PortalKind.CLIENT.value,
    "COMPANY": PortalKind.COMPANY.value,
    "DRIVER": PortalKind.DRIVER.value,
    "INSTITUTION": PortalKind.INSTITUTION.value,
}
_PORTAL_TO_ROLE = {portal: role for role, portal in _ROLE_TO_PORTAL.items()}
_TRUE = {"1", "true", "yes", "on"}
_HARD_DENY = {
    "revoked",
    "cross_tenant",
    "ambiguous_company",
    "pending",
    "unapproved",
    "ineligible",
}


@dataclass(frozen=True, slots=True)
class PortalScope:
    """Décision d'un portail pour une personne."""

    allowed: bool
    portal: str
    company: Company | None = None
    institution: Institution | None = None
    reason: str = "denied"
    source: str = "denied"


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in _TRUE


def _config_bool(name: str) -> bool:
    from flask import current_app, has_app_context

    if has_app_context() and name in current_app.config:
        return _as_bool(current_app.config.get(name))
    return os.getenv(name, "").strip().lower() in _TRUE


def enforcement_enabled() -> bool:
    """Vrai seulement si le drapeau d'application est allumé."""
    return _config_bool("PORTAL_GRANT_ENFORCEMENT")


def observation_enabled() -> bool:
    """Vrai si les divergences doivent être journalisées."""
    return _config_bool("PORTAL_GRANT_OBSERVATION")


def _role_value(user: User) -> str:
    role = getattr(user, "role", None)
    return str(getattr(role, "value", role) or "").upper()


def _int_id(raw: object) -> int | None:
    if isinstance(raw, bool) or not isinstance(raw, int):
        return None
    if raw <= 0:
        return None
    return raw


def _positive_int(raw: object) -> int | None:
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw if raw > 0 else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _jwt_claims() -> dict[str, Any]:
    try:
        from flask_jwt_extended import get_jwt

        claims = get_jwt()
        return dict(claims) if claims else {}
    except Exception:
        return {}


def _request_context_id() -> str | None:
    if not has_request_context():
        return None
    value = (request.headers.get("X-Active-Context-Id") or "").strip()
    return value or None


def _context_portal(context_id: str | None) -> str | None:
    if not context_id or ":" not in context_id:
        return None
    prefix = context_id.split(":", 1)[0]
    if prefix in _PORTAL_TO_ROLE:
        return prefix
    return None


def _context_org_id(context_id: str | None, portal: str) -> int | None:
    if _context_portal(context_id) != portal:
        return None
    return _positive_int(context_id.split(":", 1)[1] if context_id else None)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


def _legacy_cutoff() -> datetime | None:
    from flask import current_app, has_app_context

    raw: object = None
    if has_app_context():
        raw = current_app.config.get("PORTAL_GRANT_LEGACY_CUTOFF")
    if raw in (None, ""):
        raw = os.getenv("PORTAL_GRANT_LEGACY_CUTOFF") or None
    if raw in (None, ""):
        return None
    if isinstance(raw, datetime):
        return _aware(raw)
    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return _aware(datetime.fromisoformat(text))
    except ValueError:
        logger.warning("portal_legacy_cutoff_invalid")
        return None


def _portal_rows(user_id: int, portal: str) -> list[PortalGrant]:
    try:
        return PortalGrant.query.filter_by(user_id=user_id, portal=portal).all()
    except ProgrammingError:
        db.session.rollback()
        return []


def legacy_transition_certified(user_id: int, portal: str) -> bool:
    """Vrai si ce compte est nommé dans la liste de transition, pour ce portail.

    La liste est vide par défaut. ``Company.user_id`` et le domaine e-mail
    n'y figurent jamais automatiquement.
    """
    from flask import current_app, has_app_context

    raw: object = None
    if has_app_context():
        raw = current_app.config.get("PORTAL_GRANT_LEGACY_CERTIFIED")
    if raw in (None, ""):
        raw = os.getenv("PORTAL_GRANT_LEGACY_CERTIFIED") or None
    if raw in (None, ""):
        return False
    if isinstance(raw, (set, list, tuple)):
        tokens = {str(item).strip() for item in raw}
    else:
        tokens = {part.strip() for part in str(raw).split(",") if part.strip()}
    return f"{int(user_id)}:{portal}" in tokens


def legacy_fallback_eligible(user: User, portal: str) -> bool:
    """Repli historique limité à un portail et à la cohorte de transition.

    Aucune ligne pour ce portail. Le portail client suit le rôle client.
    Un portail professionnel exige en plus une certification explicite
    ``PORTAL_GRANT_LEGACY_CERTIFIED`` (``user_id:portail``) et une création
    antérieure à ``PORTAL_GRANT_LEGACY_CUTOFF``. La date seule n'ouvre rien.
    """
    user_id = getattr(user, "id", None)
    if user_id is None or _portal_rows(int(user_id), portal):
        return False
    if portal == PortalKind.CLIENT.value and _role_value(user) == UserRole.CLIENT.value:
        return True
    if not legacy_transition_certified(int(user_id), portal):
        return False
    cutoff = _legacy_cutoff()
    created = getattr(user, "created_at", None)
    if cutoff is None or not isinstance(created, datetime):
        return False
    if _aware(created) >= cutoff:
        return False
    return _role_value(user) == _PORTAL_TO_ROLE.get(portal)


def _deny(portal: str, reason: str) -> PortalScope:
    return PortalScope(allowed=False, portal=portal, reason=reason, source="denied")


def _companies_owned(user_id: int) -> list[Company]:
    return Company.query.filter_by(user_id=user_id).limit(2).all()


def _presented_company_id(
    portal: str,
    claimed_company_id: int | None,
    *,
    use_request_claim: bool,
) -> int | None:
    if claimed_company_id is not None:
        return claimed_company_id
    if not use_request_claim or portal != PortalKind.COMPANY.value:
        return None
    header = _context_org_id(_request_context_id(), portal)
    if header is not None:
        return header
    return _positive_int(_jwt_claims().get("company_id"))


def _presented_institution_id(
    portal: str,
    claimed_institution_id: int | None,
    *,
    use_request_claim: bool,
) -> int | None:
    if claimed_institution_id is not None:
        return claimed_institution_id
    if not use_request_claim or portal != PortalKind.INSTITUTION.value:
        return None
    header = _context_org_id(_request_context_id(), portal)
    if header is not None:
        return header
    return _positive_int(_jwt_claims().get("institution_id"))


def _historical_scope(
    user: User,
    portal: str,
    claimed_company_id: int | None,
    claimed_institution_id: int | None,
    *,
    require_company_approval: bool,
) -> PortalScope:
    role = _role_value(user)
    if portal == PortalKind.CLIENT.value:
        if role != UserRole.CLIENT.value:
            return _deny(portal, "role_mismatch")
        return PortalScope(
            allowed=True,
            portal=portal,
            reason="legacy_role",
            source="legacy",
        )
    if portal == PortalKind.COMPANY.value:
        if role != UserRole.COMPANY.value:
            return _deny(portal, "role_mismatch")
        owned = _companies_owned(int(user.id))
        if len(owned) > 1:
            return _deny(portal, "ambiguous_company")
        if len(owned) == 0:
            return _deny(portal, "no_organization")
        company = owned[0]
        if claimed_company_id is not None and int(claimed_company_id) != int(company.id):
            return _deny(portal, "cross_tenant")
        if require_company_approval and not bool(company.is_approved):
            return _deny(portal, "unapproved")
        return PortalScope(
            allowed=True,
            portal=portal,
            company=company,
            reason="legacy_role",
            source="legacy",
        )
    if portal == PortalKind.DRIVER.value:
        if role not in {UserRole.DRIVER.value, UserRole.COMPANY.value}:
            return _deny(portal, "role_mismatch")
        driver = getattr(user, "driver", None)
        if driver is None:
            driver = Driver.query.filter_by(user_id=user.id).first()
        if driver is None or driver.company_id is None:
            return _deny(portal, "no_organization")
        if claimed_company_id is not None and int(claimed_company_id) != int(
            driver.company_id
        ):
            return _deny(portal, "cross_tenant")
        company = Company.query.get(int(driver.company_id))
        return PortalScope(
            allowed=True,
            portal=portal,
            company=company,
            reason="legacy_role",
            source="legacy",
        )
    if portal == PortalKind.INSTITUTION.value:
        if role != UserRole.INSTITUTION.value or not getattr(user, "institution_id", None):
            return _deny(portal, "role_mismatch")
        institution_id = int(user.institution_id)
        if (
            claimed_institution_id is not None
            and int(claimed_institution_id) != institution_id
        ):
            return _deny(portal, "cross_tenant")
        institution = Institution.query.get(institution_id)
        if institution is None:
            return _deny(portal, "no_organization")
        return PortalScope(
            allowed=True,
            portal=portal,
            institution=institution,
            reason="legacy_role",
            source="legacy",
        )
    return _deny(portal, "role_mismatch")


def _approved(company_id: int | None) -> Company | None:
    if company_id is None:
        return None
    company = Company.query.get(int(company_id))
    if company is None or not bool(getattr(company, "is_approved", False)):
        return None
    return company


def _grant_scope(
    user: User,
    portal: str,
    claimed_company_id: int | None,
    claimed_institution_id: int | None,
) -> PortalScope:
    rows = _portal_rows(int(user.id), portal)
    if not rows:
        if not legacy_fallback_eligible(user, portal):
            return _deny(portal, "ineligible")
        return _historical_scope(
            user,
            portal,
            claimed_company_id,
            claimed_institution_id,
            require_company_approval=True,
        )

    revoked_companies = {
        int(row.company_id)
        for row in rows
        if row.status == PortalGrantStatus.REVOKED.value and row.company_id
    }
    revoked_institutions = {
        int(row.institution_id)
        for row in rows
        if row.status == PortalGrantStatus.REVOKED.value and row.institution_id
    }
    if portal == PortalKind.CLIENT.value and any(
        row.status == PortalGrantStatus.REVOKED.value for row in rows
    ):
        return _deny(portal, "revoked")
    if portal == PortalKind.COMPANY.value and claimed_company_id in revoked_companies:
        return _deny(portal, "revoked")
    if (
        portal == PortalKind.INSTITUTION.value
        and claimed_institution_id in revoked_institutions
    ):
        return _deny(portal, "revoked")

    active = [row for row in rows if row.status == PortalGrantStatus.ACTIVE.value]
    pending = [
        row for row in rows if row.status == PortalGrantStatus.PENDING_ACTIVATION.value
    ]
    if not active:
        if pending:
            return _deny(portal, "pending")
        return _deny(portal, "revoked")

    if portal == PortalKind.CLIENT.value:
        return PortalScope(
            allowed=True,
            portal=portal,
            reason="active_grant",
            source="grant",
        )
    if portal == PortalKind.COMPANY.value:
        chosen = active
        if claimed_company_id is not None:
            chosen = [
                row for row in active if int(row.company_id or 0) == claimed_company_id
            ]
            if not chosen:
                return _deny(portal, "cross_tenant")
        elif len(active) != 1:
            return _deny(portal, "ambiguous_company")
        company = _approved(chosen[0].company_id)
        if company is None:
            return _deny(portal, "unapproved")
        return PortalScope(
            allowed=True,
            portal=portal,
            company=company,
            reason="active_grant",
            source="grant",
        )
    if portal == PortalKind.DRIVER.value:
        driver = getattr(user, "driver", None)
        if driver is None:
            driver = Driver.query.filter_by(user_id=user.id).first()
        if driver is None or driver.company_id is None:
            return _deny(portal, "no_organization")
        if claimed_company_id is not None and claimed_company_id != int(
            driver.company_id
        ):
            return _deny(portal, "cross_tenant")
        if int(driver.company_id) in revoked_companies and not any(
            row.company_id == driver.company_id for row in active
        ):
            return _deny(portal, "revoked")
        match = next(
            (row for row in active if row.company_id == driver.company_id),
            None,
        )
        if match is None:
            return _deny(portal, "cross_tenant")
        company = Company.query.get(int(driver.company_id))
        return PortalScope(
            allowed=True,
            portal=portal,
            company=company,
            reason="active_grant",
            source="grant",
        )
    if portal == PortalKind.INSTITUTION.value:
        chosen = active
        if claimed_institution_id is not None:
            chosen = [
                row
                for row in active
                if int(row.institution_id or 0) == claimed_institution_id
            ]
            if not chosen:
                return _deny(portal, "cross_tenant")
        elif len(active) != 1:
            return _deny(portal, "ambiguous_company")
        institution = Institution.query.get(chosen[0].institution_id)
        if institution is None:
            return _deny(portal, "no_organization")
        return PortalScope(
            allowed=True,
            portal=portal,
            institution=institution,
            reason="active_grant",
            source="grant",
        )
    return _deny(portal, "role_mismatch")


def _revoked_overlay(
    user: User,
    historical: PortalScope,
    claimed_company_id: int | None,
    claimed_institution_id: int | None,
) -> PortalScope:
    """Une révocation reste un refus, même si le drapeau est éteint."""
    rows = _portal_rows(int(user.id), historical.portal)
    revoked = [row for row in rows if row.status == PortalGrantStatus.REVOKED.value]
    if not revoked:
        return historical
    if historical.portal == PortalKind.CLIENT.value:
        return _deny(historical.portal, "revoked")
    org_id = claimed_company_id
    if historical.company is not None:
        org_id = int(historical.company.id)
    if historical.portal in {PortalKind.COMPANY.value, PortalKind.DRIVER.value}:
        if org_id is not None and any(row.company_id == org_id for row in revoked):
            return _deny(historical.portal, "revoked")
        return historical
    inst_id = claimed_institution_id
    if historical.institution is not None:
        inst_id = int(historical.institution.id)
    if inst_id is not None and any(row.institution_id == inst_id for row in revoked):
        return _deny(historical.portal, "revoked")
    return historical


def _org_id(scope: PortalScope) -> int | None:
    if scope.company is not None:
        return int(scope.company.id)
    if scope.institution is not None:
        return int(scope.institution.id)
    return None


def _observe(
    user: User,
    portal: str,
    historical: PortalScope,
    future: PortalScope,
    decision: PortalScope,
) -> None:
    if not observation_enabled():
        return
    if (
        historical.allowed == future.allowed
        and _org_id(historical) == _org_id(future)
    ):
        return
    logger.info(
        "portal_scope_divergence user_id=%s portal=%s historical=%s future=%s "
        "organization_id=%s reason=%s",
        getattr(user, "id", None),
        portal,
        historical.reason if historical.allowed else f"deny:{historical.reason}",
        future.reason if future.allowed else f"deny:{future.reason}",
        _org_id(decision),
        decision.reason,
    )


def resolve_portal_scope(
    user: User,
    portal: str,
    *,
    claimed_company_id: int | None = None,
    claimed_institution_id: int | None = None,
    use_request_claim: bool = True,
) -> PortalScope:
    """Décide l'accès à un portail sans accorder un droit absent."""
    if _int_id(getattr(user, "id", None)) is None:
        return _deny(portal, "no_organization")
    if _role_value(user) == UserRole.ADMIN.value:
        return _deny(portal, "role_mismatch")
    company_claim = _presented_company_id(
        portal,
        _positive_int(claimed_company_id),
        use_request_claim=use_request_claim,
    )
    institution_claim = _presented_institution_id(
        portal,
        _positive_int(claimed_institution_id),
        use_request_claim=use_request_claim,
    )
    historical = _historical_scope(
        user,
        portal,
        company_claim,
        institution_claim,
        require_company_approval=False,
    )
    future = _grant_scope(user, portal, company_claim, institution_claim)
    if enforcement_enabled():
        decision = future
    else:
        decision = _revoked_overlay(
            user, historical, company_claim, institution_claim
        )
    _observe(user, portal, historical, future, decision)
    return decision


def portal_company_allowed(
    user: User,
    portal: str,
    company_id: int,
) -> bool:
    """Vrai si ce portail autorise exactement cette société."""
    scope = resolve_portal_scope(
        user,
        portal,
        claimed_company_id=int(company_id),
        use_request_claim=False,
    )
    return bool(
        scope.allowed
        and scope.company is not None
        and int(scope.company.id) == int(company_id)
    )


def authorized_company_id(user: User) -> int | None:
    """Société autorisée pour ce compte. ``User.company_id`` n'existe pas."""
    company_scope = resolve_portal_scope(user, "company", use_request_claim=False)
    if company_scope.allowed and company_scope.company is not None:
        return int(company_scope.company.id)
    driver_scope = resolve_portal_scope(user, "driver", use_request_claim=False)
    if driver_scope.allowed and driver_scope.company is not None:
        return int(driver_scope.company.id)
    return None
