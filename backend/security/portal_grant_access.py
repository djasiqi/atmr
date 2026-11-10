"""Contrôle progressif des habilitations multiportails.

Le drapeau ``PORTAL_GRANT_ENFORCEMENT`` est éteint par défaut. La décision
d'accès est dans ``security.portal_scope`` : un portail à la fois, sans
prendre la première société, et sans rouvrir une révocation quand le drapeau
repasse à éteint. Un droit actif n'ouvre un portail que si le drapeau est
allumé. Le repli professionnel exige une certification explicite
``PORTAL_GRANT_LEGACY_CERTIFIED`` et ``PORTAL_GRANT_LEGACY_CUTOFF``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from flask import has_request_context, request

from ext import db
from models import Company, Driver, Institution, User, UserRole
from models.client import Client
from models.enums import PortalGrantStatus, PortalKind
from models.portal_grant import PortalGrant

logger = logging.getLogger(__name__)

_ROLE_TO_PORTAL = {
    "CLIENT": PortalKind.CLIENT.value,
    "COMPANY": PortalKind.COMPANY.value,
    "DRIVER": PortalKind.DRIVER.value,
    "INSTITUTION": PortalKind.INSTITUTION.value,
}
_PORTAL_TO_ROLE_CLAIM = {portal: role for role, portal in _ROLE_TO_PORTAL.items()}


def enforcement_enabled() -> bool:
    """Indique si les gardiens doivent appliquer ``portal_grant``."""
    from security.portal_scope import enforcement_enabled as scope_enabled

    return scope_enabled()


def _role_value(user: User) -> str:
    role = getattr(user, "role", None)
    return str(getattr(role, "value", role) or "").upper()


def grants_are_authoritative(user: User) -> bool:
    """Vrai seulement si le drapeau est allumé et qu'une ligne existe."""
    if _role_value(user) == UserRole.ADMIN.value:
        return False
    if not enforcement_enabled():
        return False
    user_id = getattr(user, "id", None)
    if user_id is None:
        return False
    return (
        db.session.query(PortalGrant.id).filter_by(user_id=user_id).first()
        is not None
    )


def _active_grants(user_id: int, portal: str | None = None) -> list[PortalGrant]:
    query = PortalGrant.query.filter_by(
        user_id=user_id,
        status=PortalGrantStatus.ACTIVE.value,
    )
    if portal is not None:
        query = query.filter_by(portal=portal)
    return query.all()


def _request_context_id() -> str | None:
    if not has_request_context():
        return None
    value = (request.headers.get("X-Active-Context-Id") or "").strip()
    return value or None


def _jwt_claims() -> dict[str, Any]:
    try:
        from flask_jwt_extended import get_jwt

        claims = get_jwt()
        return dict(claims) if claims else {}
    except Exception:
        return {}


def _context_portal(context_id: str | None) -> str | None:
    if not context_id or ":" not in context_id:
        return None
    prefix = context_id.split(":", 1)[0]
    if prefix in _PORTAL_TO_ROLE_CLAIM:
        return prefix
    return None


def _positive_int(raw: object) -> int | None:
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def _company_id_from_context(context_id: str | None) -> int | None:
    if _context_portal(context_id) != PortalKind.COMPANY.value:
        return None
    return _positive_int(context_id.split(":", 1)[1] if context_id else None)


def _driver_id_from_context(context_id: str | None) -> int | None:
    if _context_portal(context_id) != PortalKind.DRIVER.value:
        return None
    return _positive_int(context_id.split(":", 1)[1] if context_id else None)


def _institution_id_from_context(context_id: str | None) -> int | None:
    if _context_portal(context_id) != PortalKind.INSTITUTION.value:
        return None
    return _positive_int(context_id.split(":", 1)[1] if context_id else None)


def _approved_company(company_id: int) -> Company | None:
    company = Company.query.get(company_id)
    if company is None or not bool(getattr(company, "is_approved", False)):
        return None
    return company


def _portal_explicit(portal: str, context_id: str | None, *, multiple: bool) -> bool:
    if _context_portal(context_id) == portal:
        return True
    claim = str(_jwt_claims().get("role") or "").upper()
    if _ROLE_TO_PORTAL.get(claim) == portal:
        return True
    return not multiple


def _active_portal_count(user_id: int) -> int:
    rows = (
        db.session.query(PortalGrant.portal)
        .filter_by(user_id=user_id, status=PortalGrantStatus.ACTIVE.value)
        .distinct()
        .all()
    )
    return len(rows)


def _select_company_grant(
    user: User, context_id: str | None
) -> PortalGrant | None:
    grants = _active_grants(user.id, PortalKind.COMPANY.value)
    if not grants:
        return None
    header_id = _company_id_from_context(context_id)
    if header_id is not None:
        match = next((row for row in grants if row.company_id == header_id), None)
        if match is None or _approved_company(match.company_id) is None:
            return None
        return match
    jwt_id = _positive_int(_jwt_claims().get("company_id"))
    if jwt_id is not None:
        match = next((row for row in grants if row.company_id == jwt_id), None)
        if match is None or _approved_company(match.company_id) is None:
            return None
        return match
    if not _portal_explicit(
        PortalKind.COMPANY.value,
        context_id,
        multiple=_active_portal_count(user.id) > 1,
    ):
        return None
    approved = [row for row in grants if _approved_company(row.company_id)]
    if len(approved) == 1:
        return approved[0]
    return None


def _select_driver_grant(user: User, context_id: str | None) -> PortalGrant | None:
    driver = getattr(user, "driver", None)
    if driver is None:
        driver = Driver.query.filter_by(user_id=user.id).first()
    if driver is None or driver.company_id is None:
        return None
    asked = _driver_id_from_context(context_id)
    if asked is not None and asked != driver.id:
        return None
    if not _portal_explicit(
        PortalKind.DRIVER.value,
        context_id,
        multiple=_active_portal_count(user.id) > 1,
    ):
        return None
    grants = _active_grants(user.id, PortalKind.DRIVER.value)
    return next(
        (row for row in grants if row.company_id == driver.company_id),
        None,
    )


def _select_institution_grant(
    user: User, context_id: str | None
) -> PortalGrant | None:
    grants = _active_grants(user.id, PortalKind.INSTITUTION.value)
    if not grants:
        return None
    header_id = _institution_id_from_context(context_id)
    if header_id is not None:
        return next(
            (row for row in grants if row.institution_id == header_id),
            None,
        )
    jwt_id = _positive_int(_jwt_claims().get("institution_id"))
    if jwt_id is not None:
        return next((row for row in grants if row.institution_id == jwt_id), None)
    if not _portal_explicit(
        PortalKind.INSTITUTION.value,
        context_id,
        multiple=_active_portal_count(user.id) > 1,
    ):
        return None
    if len(grants) == 1:
        return grants[0]
    return None


def _select_client_grant(user: User, context_id: str | None) -> PortalGrant | None:
    if _context_portal(context_id) not in (None, PortalKind.CLIENT.value):
        return None
    if not _portal_explicit(
        PortalKind.CLIENT.value,
        context_id,
        multiple=_active_portal_count(user.id) > 1,
    ):
        return None
    rows = _active_grants(user.id, PortalKind.CLIENT.value)
    return rows[0] if rows else None


def _portal_allowed(user: User, portal: str, context_id: str | None) -> bool:
    if portal == PortalKind.CLIENT.value:
        return _select_client_grant(user, context_id) is not None
    if portal == PortalKind.COMPANY.value:
        return _select_company_grant(user, context_id) is not None
    if portal == PortalKind.DRIVER.value:
        return _select_driver_grant(user, context_id) is not None
    if portal == PortalKind.INSTITUTION.value:
        return _select_institution_grant(user, context_id) is not None
    return False


def role_required_grant_gate(user: User, allowed_roles: list[object]) -> str:
    """``legacy``, ``allow``, ``deny`` ou ``driver_context_required``.

    La décision est prise portail par portail. Une révocation refuse ce
    portail même si le drapeau est éteint. Un droit actif n'est honoré
    que si le drapeau est allumé.
    """
    from security.portal_scope import resolve_portal_scope

    allowed: list[str] = []
    allows_admin = False
    for role in allowed_roles:
        value = str(getattr(role, "value", role) or "").upper()
        if value == UserRole.ADMIN.value:
            allows_admin = True
        portal = _ROLE_TO_PORTAL.get(value)
        if portal:
            allowed.append(portal)
    if allows_admin and _role_value(user) == UserRole.ADMIN.value:
        return "allow"
    if not allowed:
        return "legacy"
    scopes = [resolve_portal_scope(user, portal) for portal in allowed]
    if any(scope.allowed and scope.source == "grant" for scope in scopes):
        return "allow"
    context_portal = _context_portal(_request_context_id())
    if context_portal in allowed:
        scoped = next(scope for scope in scopes if scope.portal == context_portal)
        if not scoped.allowed and scoped.reason in {"revoked", "cross_tenant"}:
            return "deny"
    if any(scope.allowed and scope.source == "legacy" for scope in scopes):
        return "legacy"
    if any(
        scope.reason == "revoked" and _role_value(user) == _PORTAL_TO_ROLE_CLAIM.get(scope.portal)
        for scope in scopes
    ):
        return "deny"
    if enforcement_enabled():
        if any(scope.reason == "context_required" for scope in scopes):
            return "driver_context_required"
        return "deny"
    token_portal = _ROLE_TO_PORTAL.get(str(_jwt_claims().get("role") or "").upper())
    if token_portal in allowed:
        scoped = next(scope for scope in scopes if scope.portal == token_portal)
        if not scoped.allowed and scoped.reason in {
            "revoked",
            "cross_tenant",
            "ambiguous_company",
        }:
            return "deny"
    return "legacy"


def require_role_grant_gate(user: User, allowed_roles: list[UserRole]) -> str:
    """Même contrat que ``role_required_grant_gate`` pour le service central."""
    return role_required_grant_gate(user, list(allowed_roles))


def authoritative_company(user: User) -> tuple[str, Company | None]:
    """``legacy``, ``allow`` avec l'entreprise validée, ou ``deny``."""
    from security.portal_scope import resolve_portal_scope

    scope = resolve_portal_scope(user, PortalKind.COMPANY.value)
    if scope.allowed and scope.source == "grant":
        return "allow", scope.company
    if scope.allowed and scope.source == "legacy":
        return "legacy", scope.company
    if not enforcement_enabled() and scope.reason in {
        "no_organization",
        "role_mismatch",
    }:
        return "legacy", None
    return "deny", None


def company_for_user(user: User) -> Company | None:
    """Entreprise du périmètre validé. Jamais la première société trouvée."""
    _decision, company = authoritative_company(user)
    return company


def authoritative_driver(user: User) -> tuple[str, Driver | None]:
    from security.portal_scope import resolve_portal_scope

    scope = resolve_portal_scope(user, PortalKind.DRIVER.value)
    if not scope.allowed:
        if not enforcement_enabled() and scope.reason in {
            "no_organization",
            "role_mismatch",
        }:
            return "legacy", None
        return "deny", None
    driver = getattr(user, "driver", None)
    if driver is None:
        driver = Driver.query.filter_by(user_id=user.id).first()
    if driver is None:
        return "deny", None
    if scope.source == "grant":
        return "allow", driver
    return "legacy", driver


def authoritative_institution(user: User) -> tuple[str, Institution | None]:
    from security.portal_scope import resolve_portal_scope

    scope = resolve_portal_scope(user, PortalKind.INSTITUTION.value)
    if scope.allowed and scope.source == "grant" and scope.institution is not None:
        return "allow", scope.institution
    if scope.allowed and scope.source == "legacy":
        return "legacy", scope.institution
    if not enforcement_enabled() and scope.reason in {
        "no_organization",
        "role_mismatch",
    }:
        return "legacy", None
    return "deny", None


def institution_role_from_grant(user: User) -> str | None:
    """Rôle interne servi par le droit actif. ``None`` si le drapeau ne s'applique pas."""
    if not grants_are_authoritative(user):
        return None
    grant = _select_institution_grant(user, _request_context_id())
    if grant is None:
        return None
    return grant.internal_role


def company_access_decision(
    user: User, company_id: int | None
) -> bool | None:
    """Vrai si le portail entreprise autorise exactement cette société."""
    if company_id is None:
        return None
    from security.portal_scope import portal_company_allowed

    return portal_company_allowed(user, PortalKind.COMPANY.value, int(company_id))


def transport_company_access(user: User, company_id: int | None) -> bool | None:
    """Accès entreprise ou chauffeur à une société, via le périmètre validé."""
    if company_id is None:
        return None
    if company_access_decision(user, company_id) is True:
        return True
    from security.portal_scope import portal_company_allowed

    return portal_company_allowed(user, PortalKind.DRIVER.value, int(company_id))


def institution_access_decision(
    user: User, institution_id: int | None
) -> bool | None:
    if institution_id is None or not grants_are_authoritative(user):
        return None
    return any(
        row.institution_id == int(institution_id)
        for row in _active_grants(user.id, PortalKind.INSTITUTION.value)
    )


def booking_access_decision(
    booking: Any, user: User
) -> tuple[bool, tuple[dict[str, str], int] | None] | None:
    """``None`` si la règle historique du portail doit encore décider."""
    if not enforcement_enabled():
        return None
    from security.portal_scope import legacy_fallback_eligible

    error = ({"error": "Accès non autorisé"}, 403)
    company_id = getattr(booking, "company_id", None)
    if company_access_decision(user, company_id) is True:
        return True, None
    if legacy_fallback_eligible(user, PortalKind.CLIENT.value) or _active_grants(
        user.id, PortalKind.CLIENT.value
    ):
        client = Client.query.filter_by(user_id=user.id).first()
        if client is not None and client.id == getattr(booking, "client_id", None):
            return True, None
    if legacy_fallback_eligible(user, PortalKind.DRIVER.value) or _active_grants(
        user.id, PortalKind.DRIVER.value
    ):
        driver = Driver.query.filter_by(user_id=user.id).first()
        if driver is not None:
            from models import Assignment

            assigned = Assignment.query.filter_by(
                driver_id=driver.id,
                booking_id=getattr(booking, "id", None),
            ).first()
            if assigned is not None and (
                legacy_fallback_eligible(user, PortalKind.DRIVER.value)
                or any(
                    row.company_id == driver.company_id
                    for row in _active_grants(user.id, PortalKind.DRIVER.value)
                )
            ):
                return True, None
    if legacy_fallback_eligible(user, _ROLE_TO_PORTAL.get(_role_value(user), "")):
        return None
    return False, error


def contexts_from_active_grants(user: User) -> list[dict[str, object]] | None:
    """Contextes de bascule, ou ``None`` pour conserver le calcul historique."""
    if not enforcement_enabled() or not grants_are_authoritative(user):
        return None
    from routes.auth import _permissions_for_context
    from security.portal_scope import legacy_fallback_eligible

    contexts: list[dict[str, object]] = []
    active = _active_grants(user.id)
    company_ids = {
        row.company_id
        for row in active
        if row.portal == PortalKind.COMPANY.value and row.company_id
    }
    driver = getattr(user, "driver", None)
    if driver is None:
        driver = Driver.query.filter_by(user_id=user.id).first()

    def add(
        *,
        context_id: str,
        context_type: str,
        label: str,
        organization_id: int | None,
        organization_name: str | None,
        allow_switch: bool,
    ) -> None:
        contexts.append(
            {
                "context_id": context_id,
                "context_type": context_type,
                "label": label,
                "organization_id": organization_id,
                "organization_name": organization_name,
                "permissions": _permissions_for_context(context_type),
                "is_default": False,
                "allow_mobile_context_switch": allow_switch,
            }
        )

    for grant in active:
        if grant.portal == PortalKind.CLIENT.value:
            add(
                context_id="client:self",
                context_type="client",
                label="Compte client",
                organization_id=None,
                organization_name=None,
                allow_switch=False,
            )
        elif grant.portal == PortalKind.COMPANY.value and grant.company_id:
            company = _approved_company(grant.company_id)
            if company is None:
                continue
            add(
                context_id=f"company:{company.id}",
                context_type="company",
                label="Espace entreprise",
                organization_id=company.id,
                organization_name=getattr(company, "name", None),
                allow_switch=True,
            )
        elif grant.portal == PortalKind.DRIVER.value and driver is not None:
            if grant.company_id != driver.company_id:
                continue
            company = Company.query.get(driver.company_id)
            add(
                context_id=f"driver:{driver.id}",
                context_type="driver",
                label="Espace chauffeur",
                organization_id=driver.company_id,
                organization_name=getattr(company, "name", None),
                allow_switch=driver.company_id in company_ids,
            )
        elif grant.portal == PortalKind.INSTITUTION.value and grant.institution_id:
            institution = Institution.query.get(grant.institution_id)
            add(
                context_id=f"institution:{grant.institution_id}",
                context_type="institution",
                label="Espace institution",
                organization_id=grant.institution_id,
                organization_name=getattr(institution, "name", None),
                allow_switch=False,
            )
    for context in contexts:
        if context["context_type"] == "client":
            context["is_default"] = True
            break
    else:
        if contexts:
            contexts[0]["is_default"] = True
    if legacy_fallback_eligible(user, PortalKind.CLIENT.value) and not any(
        context["context_type"] == "client" for context in contexts
    ):
        add(
            context_id="client:self",
            context_type="client",
            label="Compte client",
            organization_id=None,
            organization_name=None,
            allow_switch=False,
        )
        for context in contexts:
            context["is_default"] = context["context_type"] == "client"
    return contexts


def adjust_refresh_access_claims(
    user: User,
    claims: dict[str, Any],
    *,
    active_context_id: str | None,
) -> dict[str, Any] | None:
    """Recalcule le claim de rôle. ``None`` refuse le rafraîchissement.

    Un contexte qui vise un droit absent, révoqué ou d'une autre organisation
    ne réémet pas ce rôle. La révocation est appliquée même drapeau éteint.
    """
    from security.portal_scope import resolve_portal_scope

    portal = _context_portal(active_context_id)
    if portal is None:
        portal = _ROLE_TO_PORTAL.get(str(claims.get("role") or "").upper())
    if portal is None:
        return claims
    scope = resolve_portal_scope(
        user,
        portal,
        claimed_company_id=_positive_int(claims.get("company_id")),
        claimed_institution_id=_positive_int(claims.get("institution_id")),
        use_request_claim=False,
    )
    if scope.allowed:
        if portal == PortalKind.COMPANY.value and scope.company is not None:
            updated = dict(claims)
            updated["role"] = _PORTAL_TO_ROLE_CLAIM[portal]
            updated["company_id"] = scope.company.id
            return updated
        if scope.reason in {"revoked", "cross_tenant", "ambiguous_company"}:
            return None
        return claims
    if scope.reason not in {"revoked", "cross_tenant", "ambiguous_company"}:
        if not enforcement_enabled():
            return claims
        return None
    if portal != PortalKind.CLIENT.value:
        client_scope = resolve_portal_scope(
            user,
            PortalKind.CLIENT.value,
            use_request_claim=False,
        )
        if client_scope.allowed:
            updated = dict(claims)
            updated["role"] = "CLIENT"
            updated.pop("company_id", None)
            updated.pop("institution_id", None)
            updated.pop("institution_role", None)
            return updated
    return None


def revoke_portal_grant(grant: PortalGrant) -> None:
    """Clôt une habilitation. Ne réactive jamais la ligne plus tard.

    Les sockets de ce processus sont coupées tout de suite. Les autres
    processus ne sont prévenus qu'après le commit, quand la ligne est visible.
    """
    grant.status = PortalGrantStatus.REVOKED.value
    grant.revoked_at = datetime.now(UTC)
    try:
        from sockets.chat import drop_open_sockets_for_grant

        drop_open_sockets_for_grant(grant)
    except Exception:
        logger.exception("portal_grant_socket_drop_failed user_id=%s", grant.user_id)
    _schedule_revocation_publish(grant)


def _schedule_revocation_publish(grant: PortalGrant) -> None:
    """Publie la révocation au commit, pas pendant la transaction."""
    if grant.id is None or not isinstance(grant.user_id, int):
        return
    payload = {
        "user_id": grant.user_id,
        "portal": str(grant.portal or ""),
        "company_id": grant.company_id,
        "institution_id": grant.institution_id,
    }
    grant_id = int(grant.id)
    try:
        from sqlalchemy import event

        session = db.session()
    except Exception:
        logger.exception(
            "portal_grant_revocation_schedule_failed user_id=%s", grant.user_id
        )
        return

    def _after_commit(active_session: Any) -> None:
        row = active_session.get(PortalGrant, grant_id)
        if row is None or row.status != PortalGrantStatus.REVOKED.value:
            return
        from security.portal_grant_socket_bus import publish_portal_revocation_payload

        publish_portal_revocation_payload(payload)

    event.listen(session, "after_commit", _after_commit, once=True)


def prepare_legacy_portal_grants(
    *,
    apply: bool = False,
    user_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Prépare les droits historiques sans trancher un rattachement ambigu.

    ``apply=False`` ne écrit rien. Une ligne déjà présente, y compris révoquée,
    n'est jamais recréée ni réactivée. ``User.role`` n'est pas modifié.
    """
    companies_by_user: dict[int, list[int]] = defaultdict(list)
    for company_id, owner_id in db.session.query(Company.id, Company.user_id).all():
        if owner_id is not None:
            companies_by_user[int(owner_id)].append(int(company_id))
    drivers_by_user = {
        int(owner_id): int(company_id)
        for owner_id, company_id in db.session.query(
            Driver.user_id, Driver.company_id
        ).all()
        if owner_id is not None and company_id is not None
    }
    client_user_ids = {
        int(owner_id)
        for (owner_id,) in db.session.query(Client.user_id).distinct().all()
        if owner_id is not None
    }

    query = User.query
    if user_ids is not None:
        query = query.filter(User.id.in_(user_ids))
    existing_keys = {
        (
            int(owner_id),
            portal,
            company_id,
            institution_id,
        )
        for owner_id, portal, company_id, institution_id in db.session.query(
            PortalGrant.user_id,
            PortalGrant.portal,
            PortalGrant.company_id,
            PortalGrant.institution_id,
        ).all()
    }
    report: dict[str, Any] = {
        "apply": apply,
        "created": [],
        "skipped_existing": [],
        "ambiguous": [],
        "missing": [],
        "admins_ignored": 0,
    }
    for user in query.all():
        _classify_user(
            user,
            companies_by_user,
            drivers_by_user,
            client_user_ids,
            existing_keys,
            report,
            apply=apply,
        )
    if apply:
        db.session.flush()
    return report


def _classify_user(
    user: User,
    companies_by_user: dict[int, list[int]],
    drivers_by_user: dict[int, int],
    client_user_ids: set[int],
    existing_keys: set[tuple[int, str, int | None, int | None]],
    report: dict[str, Any],
    *,
    apply: bool,
) -> None:
    role = _role_value(user)
    owned = companies_by_user.get(user.id, [])
    if role == UserRole.ADMIN.value:
        report["admins_ignored"] += 1
        if owned:
            report["ambiguous"].append(
                _ambiguous(user, "societe_sur_role_non_entreprise", owned)
            )
        return
    if role == UserRole.COMPANY.value:
        if len(owned) == 1:
            _queue_grant(
                report,
                existing_keys=existing_keys,
                apply=apply,
                user=user,
                portal=PortalKind.COMPANY.value,
                company_id=owned[0],
                institution_id=None,
                internal_role=None,
            )
        elif len(owned) == 0:
            report["missing"].append(
                _missing(user, "compte_entreprise_sans_societe")
            )
        else:
            report["ambiguous"].append(
                _ambiguous(user, "plusieurs_societes_meme_utilisateur", owned)
            )
    elif owned:
        if len(owned) > 1:
            report["ambiguous"].append(
                _ambiguous(user, "plusieurs_societes_meme_utilisateur", owned)
            )
        report["ambiguous"].append(
            _ambiguous(user, "societe_sur_role_non_entreprise", owned)
        )

    driver_company_id = drivers_by_user.get(user.id)
    if role == UserRole.DRIVER.value:
        if driver_company_id is None:
            report["missing"].append(_missing(user, "chauffeur_sans_societe"))
        else:
            _queue_grant(
                report,
                existing_keys=existing_keys,
                apply=apply,
                user=user,
                portal=PortalKind.DRIVER.value,
                company_id=driver_company_id,
                institution_id=None,
                internal_role=None,
            )
    elif role == UserRole.COMPANY.value and driver_company_id is not None:
        if len(owned) == 1 and driver_company_id == owned[0]:
            _queue_grant(
                report,
                existing_keys=existing_keys,
                apply=apply,
                user=user,
                portal=PortalKind.DRIVER.value,
                company_id=owned[0],
                institution_id=None,
                internal_role=None,
            )
        else:
            report["ambiguous"].append(
                _ambiguous(
                    user,
                    "chauffeur_societe_divergente",
                    [driver_company_id],
                )
            )
    elif driver_company_id is not None:
        report["ambiguous"].append(
            _ambiguous(
                user,
                "profil_chauffeur_hors_role",
                [driver_company_id],
            )
        )

    if role == UserRole.INSTITUTION.value:
        if user.institution_id:
            _queue_grant(
                report,
                existing_keys=existing_keys,
                apply=apply,
                user=user,
                portal=PortalKind.INSTITUTION.value,
                company_id=None,
                institution_id=int(user.institution_id),
                internal_role=user.institution_role,
            )
        else:
            report["missing"].append(_missing(user, "institution_sans_identifiant"))
    elif user.institution_id:
        report["ambiguous"].append(
            _ambiguous(
                user,
                "institution_id_sans_role",
                [int(user.institution_id)],
            )
        )

    if role == UserRole.CLIENT.value:
        if user.id in client_user_ids:
            _queue_grant(
                report,
                existing_keys=existing_keys,
                apply=apply,
                user=user,
                portal=PortalKind.CLIENT.value,
                company_id=None,
                institution_id=None,
                internal_role=None,
            )
        else:
            report["missing"].append(_missing(user, "compte_client_sans_profil"))


def _queue_grant(
    report: dict[str, Any],
    *,
    existing_keys: set[tuple[int, str, int | None, int | None]],
    apply: bool,
    user: User,
    portal: str,
    company_id: int | None,
    institution_id: int | None,
    internal_role: str | None,
) -> None:
    key = (int(user.id), portal, company_id, institution_id)
    entry = {
        "user_id": user.id,
        "portal": portal,
        "company_id": company_id,
        "institution_id": institution_id,
        "role_preserved": _role_value(user),
    }
    if key in existing_keys:
        report["skipped_existing"].append(entry)
        return
    existing_keys.add(key)
    report["created"].append(entry)
    if not apply:
        return
    db.session.add(
        PortalGrant(
            user_id=user.id,
            portal=portal,
            company_id=company_id,
            institution_id=institution_id,
            internal_role=internal_role,
            status=PortalGrantStatus.ACTIVE.value,
            granted_at=datetime.now(UTC),
        )
    )


def _ambiguous(user: User, reason: str, organization_ids: list[int]) -> dict[str, Any]:
    return {
        "user_id": user.id,
        "role": _role_value(user),
        "reason": reason,
        "organization_ids": organization_ids,
    }


def _missing(user: User, reason: str) -> dict[str, Any]:
    return {"user_id": user.id, "role": _role_value(user), "reason": reason}


def ambiguous_company_summary(report: dict[str, Any]) -> dict[str, int]:
    """Compte les motifs ambigus, sans lister les identifiants."""
    counts: dict[str, int] = defaultdict(int)
    for row in report.get("ambiguous", []):
        counts[str(row.get("reason"))] += 1
    total = sum(counts.values())
    counts["total"] = total
    return dict(counts)
