"""Transitions de rôle admin sécurisées (pas de restore de l'ancien UpdateUserRole)."""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select

from ext import db
from models.company import Company
from models.control_plane import (
    OrganizationMembership,
    PlatformOrganization,
    RoleTemplate,
)
from models.driver import Driver
from models.enums import InstitutionRole, UserRole
from models.institution import Institution
from models.user import User
from security.audit_log import AuditLog
from services.admin_authz import (
    CAP_USERS_MANAGE,
    CAP_USERS_SECURITY,
    user_has_admin_capability,
)
from services.admin_role_utils import normalized_role_value
from services.control_plane.classification import (
    CompanyProjectionKind,
    classify_company_for_control_plane,
)
from services.control_plane.projector import get_projector

logger = logging.getLogger(__name__)


class RoleTransitionError(Exception):
    """Erreur métier de transition avec code HTTP et payload."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error = error
        self.details = details or {}

    def to_response(self) -> tuple[dict[str, Any], int]:
        body: dict[str, Any] = {"message": self.message}
        if self.error:
            body["error"] = self.error
        if self.details:
            body["details"] = self.details
        return body, self.status_code


@dataclass
class RoleTransitionPreview:
    allowed: bool
    old_role: str
    new_role: str
    changes: list[str] = field(default_factory=list)
    preserved_data: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    preview_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    old_context: dict[str, Any] = field(default_factory=dict)
    new_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "old_role": self.old_role,
            "new_role": self.new_role,
            "changes": self.changes,
            "preserved_data": self.preserved_data,
            "warnings": self.warnings,
            "blockers": self.blockers,
            "preview_id": self.preview_id,
            "old_context": self.old_context,
            "new_context": self.new_context,
        }


@dataclass
class RoleTransitionResult:
    user: User
    sessions_revoked: bool
    reauthentication_required: bool
    transition_id: str
    noop: bool = False
    message: str = "Rôle mis à jour."

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "role": normalized_role_value(self.user.role),
            "sessions_revoked": self.sessions_revoked,
            "reauthentication_required": self.reauthentication_required,
            "transition_id": self.transition_id,
            "noop": self.noop,
            "user": self.user.serialize
            if hasattr(self.user, "serialize")
            else {"id": self.user.id},
        }


def _now() -> datetime:
    return datetime.now(UTC)


def _parse_role(raw: str) -> UserRole:
    key = (raw or "").strip().lower()
    upper = key.upper()
    try:
        return UserRole[upper]
    except KeyError:
        for r in UserRole:
            if str(r.value).upper() == upper or str(r.value).lower() == key:
                return r
    raise RoleTransitionError("Rôle invalide.", status_code=400, error="invalid_role")


def _driver_for_user(user: User) -> Driver | None:
    return db.session.scalar(select(Driver).where(Driver.user_id == user.id))


def _owned_companies(user_id: int) -> list[Company]:
    return list(
        db.session.scalars(select(Company).where(Company.user_id == user_id)).all()
    )


def _transport_tenants_owned(user_id: int) -> list[Company]:
    out: list[Company] = []
    for c in _owned_companies(user_id):
        decision = classify_company_for_control_plane(c)
        if decision.kind == CompanyProjectionKind.TRANSPORT_TENANT:
            out.append(c)
    return out


def _has_active_company_owner_membership(user_id: int) -> bool:
    owner_role = db.session.scalar(
        select(RoleTemplate).where(
            RoleTemplate.organization_type == "company",
            RoleTemplate.role_key == "company_owner",
        )
    )
    if owner_role is None:
        return False
    m = db.session.scalar(
        select(OrganizationMembership).where(
            OrganizationMembership.user_id == user_id,
            OrganizationMembership.role_template_id == owner_role.id,
            OrganizationMembership.membership_status == "active",
        )
    )
    return m is not None


def _is_active_company_owner(user: User) -> bool:
    if _has_active_company_owner_membership(int(user.id)):
        return True
    # Owner legacy d'un transport tenant non encore projeté
    return (
        len(_transport_tenants_owned(int(user.id))) > 0
        and normalized_role_value(user.role) == "COMPANY"
    )


def _resolve_company_for_driver(company_id: int | None) -> Company:
    if not company_id:
        raise RoleTransitionError(
            "company_id est requis pour le rôle chauffeur.",
            status_code=400,
            error="company_id_required",
        )
    company = db.session.get(Company, company_id)
    if company is None:
        raise RoleTransitionError(
            "Entreprise introuvable.",
            status_code=404,
            error="company_not_found",
        )
    decision = classify_company_for_control_plane(company)
    if decision.kind != CompanyProjectionKind.TRANSPORT_TENANT:
        raise RoleTransitionError(
            "L'entreprise n'est pas un tenant de transport admissible.",
            status_code=422,
            error="company_not_transport_tenant",
            details={"kind": decision.kind.value, "reason": decision.reason},
        )
    if bool(getattr(company, "platform_suspended", False)):
        raise RoleTransitionError(
            "L'entreprise est suspendue.",
            status_code=422,
            error="company_suspended",
        )
    return company


def _resolve_target_company_owner(user: User, company_id: int | None) -> Company:
    tenants = _transport_tenants_owned(int(user.id))
    if company_id is not None:
        company = db.session.get(Company, company_id)
        if company is None:
            raise RoleTransitionError(
                "Entreprise introuvable.",
                status_code=404,
                error="company_not_found",
            )
        if int(company.user_id or 0) != int(user.id):
            raise RoleTransitionError(
                "L'entreprise n'est pas déjà liée à ce compte comme propriétaire.",
                status_code=409,
                error="company_owner_assignment_required",
            )
        decision = classify_company_for_control_plane(company)
        if decision.kind != CompanyProjectionKind.TRANSPORT_TENANT:
            raise RoleTransitionError(
                "L'entreprise n'est pas un tenant de transport.",
                status_code=422,
                error="company_not_transport_tenant",
            )
        return company
    if len(tenants) == 1:
        return tenants[0]
    raise RoleTransitionError(
        "Attribution propriétaire entreprise requise (tenant unique ou company_id explicite).",
        status_code=409,
        error="company_owner_assignment_required",
        details={"transport_tenant_count": len(tenants)},
    )


def _current_context(user: User) -> dict[str, Any]:
    drv = _driver_for_user(user)
    return {
        "role": normalized_role_value(user.role),
        "company_id": int(drv.company_id) if drv and drv.company_id else None,
        "driver_id": int(drv.id) if drv else None,
        "institution_id": user.institution_id,
        "institution_role": user.institution_role,
    }


def _assert_expected_context(
    user: User,
    *,
    expected_current_role: str,
    expected_company_id: int | None,
    expected_institution_id: int | None,
    expected_institution_role: str | None,
) -> None:
    ctx = _current_context(user)
    exp_role = normalized_role_value(expected_current_role)
    if ctx["role"] != exp_role:
        raise RoleTransitionError(
            "Le rôle actuel a changé depuis le chargement du formulaire.",
            status_code=409,
            error="concurrent_role_mismatch",
            details={"expected": exp_role, "actual": ctx["role"]},
        )
    if expected_company_id is not None and ctx["company_id"] != int(
        expected_company_id
    ):
        raise RoleTransitionError(
            "L'entreprise chauffeur actuelle a changé.",
            status_code=409,
            error="concurrent_company_mismatch",
        )
    if expected_institution_id is not None and ctx["institution_id"] != int(
        expected_institution_id
    ):
        raise RoleTransitionError(
            "L'institution actuelle a changé.",
            status_code=409,
            error="concurrent_institution_mismatch",
        )
    if (
        expected_institution_role is not None
        and (ctx["institution_role"] or "") != expected_institution_role
    ):
        raise RoleTransitionError(
            "Le rôle institution actuel a changé.",
            status_code=409,
            error="concurrent_institution_role_mismatch",
        )


def _count_platform_admins() -> int:
    return (
        db.session.scalar(
            select(func.count()).select_from(User).where(User.role == UserRole.ADMIN)
        )
        or 0
    )


def _future_bookings_warning(driver: Driver | None) -> str | None:
    if driver is None:
        return None
    try:
        from models import Booking, BookingStatus

        count = db.session.scalar(
            select(func.count())
            .select_from(Booking)
            .where(
                Booking.driver_id == driver.id,
                Booking.status.in_(
                    (
                        BookingStatus.ASSIGNED,
                        BookingStatus.ACCEPTED,
                        BookingStatus.PENDING,
                    )
                ),
            )
        )
        if count:
            return f"{int(count)} course(s) assignée(s)/en attente potentiellement impactée(s)."
    except Exception:
        logger.debug(
            "Impossible de compter les courses futures pour preview", exc_info=True
        )
    return None


class AdminAccountRoleTransitionService:
    """Autorité unique des transitions de rôle admin."""

    def preview(
        self,
        *,
        user_id: int,
        target_role: str,
        company_id: int | None = None,
        institution_id: int | None = None,
        institution_role: str | None = None,
        expected_current_role: str | None = None,
        expected_company_id: int | None = None,
        expected_institution_id: int | None = None,
        expected_institution_role: str | None = None,
    ) -> RoleTransitionPreview:
        user = db.session.get(User, user_id)
        if user is None:
            raise RoleTransitionError(
                "Compte introuvable.", status_code=404, error="user_not_found"
            )

        if expected_current_role:
            _assert_expected_context(
                user,
                expected_current_role=expected_current_role,
                expected_company_id=expected_company_id,
                expected_institution_id=expected_institution_id,
                expected_institution_role=expected_institution_role,
            )

        old_ctx = _current_context(user)
        new_role_enum = _parse_role(target_role)
        new_role = normalized_role_value(new_role_enum)
        preview = RoleTransitionPreview(
            allowed=True,
            old_role=old_ctx["role"],
            new_role=new_role,
            old_context=old_ctx,
            new_context={
                "role": new_role,
                "company_id": company_id,
                "institution_id": institution_id,
                "institution_role": institution_role,
            },
        )

        try:
            self._validate_transition(
                user,
                new_role_enum,
                company_id=company_id,
                institution_id=institution_id,
                institution_role=institution_role,
                actor_admin_id=None,
                for_preview=True,
            )
        except RoleTransitionError as exc:
            preview.allowed = False
            preview.blockers.append(
                {"code": exc.error or "blocked", "message": exc.message, **exc.details}
            )
            return preview

        # No-op?
        if self._is_noop(
            user,
            new_role_enum,
            company_id=company_id,
            institution_id=institution_id,
            institution_role=institution_role,
        ):
            preview.changes.append("Aucun changement (même rôle et même contexte).")
            return preview

        preview.changes.append(f"Rôle {old_ctx['role']} → {new_role}")
        preview.preserved_data.append(
            "Historique et données liées conservés (pas de suppression physique)"
        )
        preview.changes.append("Révocation de toutes les sessions")

        if new_role == "DRIVER":
            company = _resolve_company_for_driver(company_id)
            preview.changes.append(f"Rattachement chauffeur à « {company.name} »")
            drv = _driver_for_user(user)
            if drv is None:
                preview.changes.append("Création du profil chauffeur")
            else:
                preview.changes.append("Réactivation / mise à jour du profil chauffeur")
            preview.changes.append("Appartenance company_driver active")
            warn = _future_bookings_warning(drv)
            if warn and old_ctx["company_id"] and company_id != old_ctx["company_id"]:
                preview.warnings.append(warn)
                preview.changes[0] = (
                    f"Transférer le chauffeur : entreprise {old_ctx['company_id']} → {company_id}"
                )

        if old_ctx["role"] == "DRIVER" and new_role == "CLIENT":
            preview.changes.append("Désactivation du profil chauffeur (conservé)")
            preview.preserved_data.append("Profil chauffeur et historique courses")

        if new_role == "INSTITUTION":
            inst = db.session.get(Institution, institution_id)
            preview.changes.append(
                f"Rattachement institution « {inst.name if inst else institution_id} »"
                f" / rôle {institution_role}"
            )

        if new_role == "COMPANY":
            co = _resolve_target_company_owner(user, company_id)
            preview.changes.append(
                f"Confirmation propriétaire transport tenant « {co.name} » (aucune création)"
            )

        if new_role == "ADMIN":
            preview.changes.append("Promotion administrateur plateforme")
            preview.warnings.append(
                "Action sensible : dual capacité manage+security requise"
            )

        return preview

    def apply(
        self,
        *,
        user_id: int,
        target_role: str,
        expected_current_role: str,
        reason: str,
        actor_admin_id: int,
        company_id: int | None = None,
        institution_id: int | None = None,
        institution_role: str | None = None,
        expected_company_id: int | None = None,
        expected_institution_id: int | None = None,
        expected_institution_role: str | None = None,
        preview_id: str | None = None,
        transition_id: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        request_id: str | None = None,
    ) -> RoleTransitionResult:
        if not reason or len(reason.strip()) < 5:
            raise RoleTransitionError(
                "Une raison d'au moins 5 caractères est requise.",
                status_code=400,
                error="reason_required",
            )

        user = db.session.execute(
            select(User).where(User.id == user_id).with_for_update()
        ).scalar_one_or_none()
        if user is None:
            raise RoleTransitionError(
                "Compte introuvable.", status_code=404, error="user_not_found"
            )

        _assert_expected_context(
            user,
            expected_current_role=expected_current_role,
            expected_company_id=expected_company_id,
            expected_institution_id=expected_institution_id,
            expected_institution_role=expected_institution_role,
        )

        new_role_enum = _parse_role(target_role)
        old_ctx = _current_context(user)
        tid = transition_id or str(uuid.uuid4())

        self._validate_transition(
            user,
            new_role_enum,
            company_id=company_id,
            institution_id=institution_id,
            institution_role=institution_role,
            actor_admin_id=actor_admin_id,
            for_preview=False,
        )

        if self._is_noop(
            user,
            new_role_enum,
            company_id=company_id,
            institution_id=institution_id,
            institution_role=institution_role,
        ):
            return RoleTransitionResult(
                user=user,
                sessions_revoked=False,
                reauthentication_required=False,
                transition_id=tid,
                noop=True,
                message="Rôle inchangé.",
            )

        old_role = old_ctx["role"]
        old_company_id = old_ctx["company_id"]
        old_institution_id = old_ctx["institution_id"]
        old_driver_id = old_ctx["driver_id"]

        # --- Mutations legacy ---
        self._apply_legacy_mutations(
            user,
            new_role_enum,
            company_id=company_id,
            institution_id=institution_id,
            institution_role=institution_role,
        )
        db.session.flush()

        # --- Projection CP ---
        get_projector().sync_user_role_transition(
            user,
            old_role=old_role,
            old_company_id=old_company_id,
            old_institution_id=old_institution_id,
            old_driver_id=old_driver_id,
        )
        db.session.flush()

        # --- Sessions ---
        from security.mobile_device_session_service import revoke_user_security_sessions

        try:
            revoke_user_security_sessions(
                user,
                reason="admin_role_transition",
                increment_token_version=True,
                fail_closed=True,
                commit_tokens=False,
            )
            sessions_revoked = True
        except Exception as exc:
            db.session.rollback()
            raise RoleTransitionError(
                "Impossible de révoquer les sessions ; transition annulée.",
                status_code=500,
                error="session_revoke_failed",
                details={"detail": str(exc)},
            ) from exc

        new_ctx = _current_context(user)
        audit = AuditLog()
        audit.user_id = actor_admin_id
        audit.user_type = "ADMIN"
        audit.action_type = "admin_role_transition"
        audit.action_category = "security"
        audit.action_details = json.dumps(
            {
                "transition_id": tid,
                "preview_id": preview_id,
                "target_user_id": user.id,
                "old_context": old_ctx,
                "new_context": new_ctx,
                "reason": reason.strip(),
                "sessions_revoked": sessions_revoked,
                "request_id": request_id,
            },
            ensure_ascii=False,
            default=str,
        )
        audit.result_status = "success"
        audit.result_message = "role_transition_applied"
        audit.ip_address = ip_address
        audit.user_agent = user_agent
        audit.resource_type = "user"
        audit.resource_id = str(user.id)
        audit.created_at = _now()
        db.session.add(audit)

        db.session.commit()
        return RoleTransitionResult(
            user=user,
            sessions_revoked=sessions_revoked,
            reauthentication_required=True,
            transition_id=tid,
        )

    def _is_noop(
        self,
        user: User,
        new_role: UserRole,
        *,
        company_id: int | None,
        institution_id: int | None,
        institution_role: str | None,
    ) -> bool:
        ctx = _current_context(user)
        if ctx["role"] != normalized_role_value(new_role):
            return False
        if normalized_role_value(new_role) == "DRIVER":
            return company_id is None or ctx["company_id"] == int(company_id)
        if normalized_role_value(new_role) == "INSTITUTION":
            same_inst = institution_id is None or ctx["institution_id"] == int(
                institution_id
            )
            same_irole = (
                institution_role is None
                or (ctx["institution_role"] or "") == institution_role
            )
            return same_inst and same_irole
        return True

    def _validate_transition(
        self,
        user: User,
        new_role: UserRole,
        *,
        company_id: int | None,
        institution_id: int | None,
        institution_role: str | None,
        actor_admin_id: int | None,
        for_preview: bool,
    ) -> None:
        new_role_s = normalized_role_value(new_role)
        old_role_s = normalized_role_value(user.role)

        # Quitter COMPANY avec ownership
        if old_role_s == "COMPANY" and new_role_s != "COMPANY":
            if _is_active_company_owner(user):
                raise RoleTransitionError(
                    "Transition hors ownership Company requiert l'assistant CP-PR3.",
                    status_code=409,
                    error="company_ownership_transition_required",
                )

        if new_role_s == "DRIVER":
            _resolve_company_for_driver(company_id)

        if new_role_s == "INSTITUTION":
            if not institution_id:
                raise RoleTransitionError(
                    "institution_id est requis.",
                    status_code=400,
                    error="institution_id_required",
                )
            if not institution_role:
                raise RoleTransitionError(
                    "institution_role est requis.",
                    status_code=400,
                    error="institution_role_required",
                )
            if institution_role not in InstitutionRole.choices():
                raise RoleTransitionError(
                    "institution_role invalide.",
                    status_code=400,
                    error="invalid_institution_role",
                )
            inst = db.session.get(Institution, institution_id)
            if inst is None:
                raise RoleTransitionError(
                    "Institution introuvable.",
                    status_code=404,
                    error="institution_not_found",
                )

        if new_role_s == "COMPANY":
            _resolve_target_company_owner(user, company_id)

        if new_role_s == "ADMIN" and not for_preview:
            if actor_admin_id is None:
                raise RoleTransitionError(
                    "Acteur admin requis.",
                    status_code=403,
                    error="actor_required",
                )
            if int(actor_admin_id) == int(user.id):
                raise RoleTransitionError(
                    "Impossible de se promouvoir ou se rétrograder soi-même.",
                    status_code=409,
                    error="self_admin_transition_forbidden",
                )
            if not user_has_admin_capability(actor_admin_id, CAP_USERS_MANAGE):
                raise RoleTransitionError(
                    "Capacité admin.users.manage requise.",
                    status_code=403,
                    error="capability_denied",
                )
            if not user_has_admin_capability(actor_admin_id, CAP_USERS_SECURITY):
                raise RoleTransitionError(
                    "Capacité admin.users.security requise pour promotion ADMIN.",
                    status_code=403,
                    error="capability_denied",
                )

        if old_role_s == "ADMIN" and new_role_s != "ADMIN":
            if actor_admin_id is not None and int(actor_admin_id) == int(user.id):
                raise RoleTransitionError(
                    "Impossible de se rétrograder soi-même.",
                    status_code=409,
                    error="self_admin_transition_forbidden",
                )
            if _count_platform_admins() <= 1:
                raise RoleTransitionError(
                    "Impossible de rétrograder le dernier administrateur plateforme.",
                    status_code=409,
                    error="last_platform_admin_protected",
                )
            if not for_preview and actor_admin_id is not None:
                if not (
                    user_has_admin_capability(actor_admin_id, CAP_USERS_MANAGE)
                    and user_has_admin_capability(actor_admin_id, CAP_USERS_SECURITY)
                ):
                    raise RoleTransitionError(
                        "Dual capacité manage+security requise pour rétrograder un ADMIN.",
                        status_code=403,
                        error="capability_denied",
                    )

    def _apply_legacy_mutations(
        self,
        user: User,
        new_role: UserRole,
        *,
        company_id: int | None,
        institution_id: int | None,
        institution_role: str | None,
    ) -> None:
        new_role_s = normalized_role_value(new_role)
        user.role = new_role  # type: ignore[assignment]

        if new_role_s == "DRIVER":
            company = _resolve_company_for_driver(company_id)
            drv = _driver_for_user(user)
            if drv is None:
                drv = Driver()
                drv.user_id = user.id
                drv.company_id = company.id
                drv.is_active = True
                drv.is_available = True
                db.session.add(drv)
            else:
                drv.company_id = company.id
                drv.is_active = True
                drv.is_available = True
            # Clear institution link if any
            user.institution_id = None
            user.institution_role = None

        elif new_role_s == "CLIENT":
            drv = _driver_for_user(user)
            if drv is not None:
                drv.is_active = False
                drv.is_available = False
            user.institution_id = None
            user.institution_role = None

        elif new_role_s == "INSTITUTION":
            assert institution_id is not None
            assert institution_role is not None
            user.institution_id = institution_id
            user.institution_role = institution_role
            drv = _driver_for_user(user)
            if drv is not None:
                drv.is_active = False
                drv.is_available = False

        elif new_role_s == "COMPANY":
            # Ownership déjà validée ; pas de create/approve/dispatch
            user.institution_id = None
            user.institution_role = None
            _resolve_target_company_owner(user, company_id)

        elif new_role_s == "ADMIN":
            # Admin plateforme : détacher toute ownership entreprise
            # (évite JWT company_id / login espace transport)
            for owned in _owned_companies(user.id):
                owned.user_id = None
                logger.info(
                    "ADMIN transition: détaché ownership company_id=%s user_id=%s",
                    owned.id,
                    user.id,
                )
            user.institution_id = None
            user.institution_role = None
            drv = _driver_for_user(user)
            if drv is not None:
                drv.is_active = False
                drv.is_available = False


def list_transport_tenants_for_picker() -> list[dict[str, Any]]:
    """Companies classées TRANSPORT_TENANT pour sélecteurs admin."""
    companies = db.session.scalars(select(Company).order_by(Company.name)).all()
    out: list[dict[str, Any]] = []
    for c in companies:
        decision = classify_company_for_control_plane(c)
        if decision.kind != CompanyProjectionKind.TRANSPORT_TENANT:
            continue
        if bool(getattr(c, "platform_suspended", False)):
            continue
        out.append(
            {
                "id": c.id,
                "name": c.name,
                "owner_user_id": c.user_id,
            }
        )
    return out
