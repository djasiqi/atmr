"""Approbation atomique d'une préinscription. Le verrou est éteint par défaut."""

from __future__ import annotations

import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta

from flask import current_app
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError

from ext import db
from models import Company, Institution, PortalGrant, User, UserRole
from models.enums import (
    InstitutionRole,
    OrganizationApplicationKind,
    OrganizationApplicationStatus,
    PortalGrantStatus,
    PortalKind,
)
from models.organization_application import OrganizationApplication
from security.portal_scope import portal_company_allowed
from services.organization_application_gate import approval_enabled
from services.organization_application_review import (
    _checks,
    _detail,
    _event,
    _locked,
    _notify,
    _persist,
    _reject_self_review,
)
from services.organization_applications import (
    ApplicationRejected,
    _issue_token,
    _usable_token,
)

logger = logging.getLogger(__name__)

_INVITE_PURPOSE = "institution_admin_invite"
_INVITE_TTL = timedelta(days=7)


def approve_application(public_id: str, actor: User) -> dict:
    """Crée ou rattache l'organisation seulement si le verrou est ouvert."""
    if not approval_enabled():
        logger.info("approbation bloquee dossier=%s", public_id)
        raise ApplicationRejected(
            "activation_bloquee",
            "L'approbation qui crée une organisation ou une habilitation "
            "n'est pas disponible.",
            status=403,
        )
    row = _locked(public_id)
    _reject_self_review(row, actor)
    if row.status == OrganizationApplicationStatus.APPROVED.value:
        detail = _detail(row)
        detail["notification"] = None
        return detail
    _require_admissible(row)
    try:
        with db.session.begin_nested():
            if row.kind == OrganizationApplicationKind.COMPANY.value:
                notification = _approve_company(row, actor)
            else:
                notification = _approve_institution(row, actor)
            _persist()
    except ApplicationRejected:
        db.session.expire_all()
        raise
    except IntegrityError as error:
        db.session.expire_all()
        raise ApplicationRejected(
            "decision_en_cours",
            "Cette approbation est déjà en cours de traitement.",
            status=409,
        ) from error
    db.session.expire(row, ["events"])
    detail = _detail(row)
    detail["notification"] = notification
    return detail


def accept_institution_invitation(raw_token: str) -> dict[str, str]:
    """Active le premier administrateur. Un autre jeton ne convient pas."""
    token = _usable_token(raw_token, _INVITE_PURPOSE)
    if token is None:
        raise ApplicationRejected("lien_invalide", "Ce lien d'activation n'est pas valable.")
    row = token.application
    grant = _institution_grant(row)
    if grant is None or grant.status == PortalGrantStatus.REVOKED.value:
        raise ApplicationRejected("lien_invalide", "Ce lien d'activation n'est pas valable.")
    if grant.status == PortalGrantStatus.ACTIVE.value and token.used_at is not None:
        return {"message": "Cette invitation est déjà acceptée."}
    user = db.session.get(User, grant.user_id)
    if user is None:
        raise ApplicationRejected("lien_invalide", "Ce lien d'activation n'est pas valable.")
    role_before = user.role
    token.used_at = datetime.now(UTC)
    grant.status = PortalGrantStatus.ACTIVE.value
    user.institution_id = grant.institution_id
    user.institution_role = InstitutionRole.ADMIN.value
    user.role = role_before
    _event(row, "institution_admin_activated", None, None, internal=False)
    _persist()
    return {"message": "Votre accès institutionnel est activé."}


def revoke_approved_grant(public_id: str, actor: User) -> dict:
    """Révoque l'habilitation issue de l'approbation. La ligne reste."""
    if not approval_enabled():
        raise ApplicationRejected(
            "activation_bloquee",
            "L'approbation qui crée une organisation ou une habilitation "
            "n'est pas disponible.",
            status=403,
        )
    row = _locked(public_id)
    _reject_self_review(row, actor)
    if row.status != OrganizationApplicationStatus.APPROVED.value:
        raise ApplicationRejected(
            "transition_refusee",
            "Seule une demande approuvée peut être révoquée.",
            status=409,
        )
    grant = _grant_for(row)
    if grant is None:
        raise ApplicationRejected("habilitation_absente", "Aucune habilitation à révoquer.")
    if grant.status != PortalGrantStatus.REVOKED.value:
        grant.status = PortalGrantStatus.REVOKED.value
        grant.revoked_at = datetime.now(UTC)
        _event(row, "grant_revoked", actor, None, internal=True)
        _persist()
    db.session.expire(row, ["events"])
    detail = _detail(row)
    detail["notification"] = None
    return detail


def _approve_company(row: OrganizationApplication, actor: User) -> dict:
    user = _applicant_user(row)
    company = _company_for(row, user)
    _ensure_grant(
        user,
        PortalKind.COMPANY.value,
        company_id=company.id,
        status=PortalGrantStatus.ACTIVE.value,
        internal_role=None,
    )
    row.matched_company_id = company.id
    row.user_id = user.id
    row.status = OrganizationApplicationStatus.APPROVED.value
    _event(row, "approved", actor, None, internal=False)
    access = portal_company_allowed(user, PortalKind.COMPANY.value, int(company.id))
    if not access:
        raise ApplicationRejected(
            "acces_non_ouvert",
            "L'approbation n'ouvre pas encore le portail de cette entreprise.",
            status=409,
        )
    return _notify(
        row,
        "company_approved",
        "Votre demande d'entreprise est approuvée. "
        "L'accès au portail suit l'habilitation enregistrée.",
    )


def _approve_institution(row: OrganizationApplication, actor: User) -> dict:
    user = _applicant_user(row)
    institution = _institution_for(row)
    _ensure_grant(
        user,
        PortalKind.INSTITUTION.value,
        institution_id=institution.id,
        status=PortalGrantStatus.PENDING_ACTIVATION.value,
        internal_role=InstitutionRole.ADMIN.value,
    )
    row.matched_institution_id = institution.id
    row.user_id = user.id
    row.status = OrganizationApplicationStatus.APPROVED.value
    _event(row, "approved", actor, None, internal=False)
    raw = _issue_token(row, _INVITE_PURPOSE, _INVITE_TTL)
    if current_app.config.get("TESTING"):
        current_app.config.setdefault("_ORG_APP_LINKS", []).append(
            {"email": row.applicant_email, "token": raw, "purpose": _INVITE_PURPOSE}
        )
    _event(row, "institution_invitation_sent", actor, None, internal=False)
    return _notify(
        row,
        "institution_invitation",
        "Votre institution est enregistrée. "
        "L'accès s'ouvre seulement après le lien d'activation du message le plus récent.",
    )


def _require_admissible(row: OrganizationApplication) -> None:
    if row.status == OrganizationApplicationStatus.EMAIL_UNVERIFIED.value:
        raise ApplicationRejected(
            "dossier_incomplet",
            "Ce dossier n'est pas complet : l'adresse e-mail n'est pas confirmée.",
        )
    if row.status == OrganizationApplicationStatus.REFUSED.value:
        raise ApplicationRejected(
            "transition_refusee",
            "Une demande refusée ne peut pas être approuvée.",
            status=409,
        )
    if row.status != OrganizationApplicationStatus.IN_REVIEW.value or row.email_verified_at is None:
        raise ApplicationRejected(
            "transition_refusee",
            "Ce dossier ne peut pas être approuvé depuis son état actuel.",
            status=409,
        )
    checks = _checks(row)
    if checks["identity"]["confirmed"] is not True or checks["organization"]["confirmed"] is not True:
        raise ApplicationRejected(
            "dossier_inadmissible",
            "L'identité et l'organisation doivent être confirmées avant l'approbation.",
        )


def _applicant_user(row: OrganizationApplication) -> User:
    user = None
    if row.user_id is not None:
        user = db.session.get(User, row.user_id)
    if user is None:
        user = User.query.filter(func.lower(User.email) == row.applicant_email.lower()).one_or_none()
    role_before = None if user is None else user.role
    if user is None:
        user = User()
        user.username = f"ins_{uuid.uuid4().hex[:12]}"
        user.email = row.applicant_email
        user.public_id = str(uuid.uuid4())
        user.role = UserRole.CLIENT
        user.set_password(secrets.token_urlsafe(24), force_change=False)
        db.session.add(user)
        db.session.flush()
        role_before = user.role
    if role_before != user.role:
        user.role = role_before
    return user


def _company_for(row: OrganizationApplication, user: User) -> Company:
    if row.matched_company_id is not None:
        company = db.session.get(Company, row.matched_company_id)
        if company is None:
            raise ApplicationRejected(
                "organisation_introuvable",
                "L'entreprise notée n'existe plus.",
                status=404,
            )
        if not company.is_approved:
            company.is_approved = True
        return company
    matches = _company_matches(row)
    if len(matches) == 1:
        raise ApplicationRejected(
            "organisation_existante",
            "Une entreprise correspond déjà. Notez-la explicitement avant d'approuver.",
            status=409,
        )
    if len(matches) > 1:
        raise ApplicationRejected(
            "organisation_ambigue",
            "Plusieurs entreprises correspondent. Choisissez-en une explicitement.",
            status=409,
        )
    company = Company()
    company.name = (row.legal_name or "")[:100]
    company.legal_name = row.legal_name
    company.uid_ide = row.ide_number
    company.address = row.street[:200]
    company.contact_email = row.applicant_email[:100]
    company.contact_phone = row.applicant_phone[:20]
    company.user_id = user.id
    company.is_approved = True
    db.session.add(company)
    db.session.flush()
    return company


def _company_matches(row: OrganizationApplication) -> list[Company]:
    filters = [func.lower(Company.name) == (row.legal_name or "").lower()]
    if row.ide_number:
        filters.append(Company.uid_ide == row.ide_number)
    return Company.query.filter(or_(*filters)).limit(2).all()


def _institution_for(row: OrganizationApplication) -> Institution:
    if row.matched_institution_id is not None:
        institution = db.session.get(Institution, row.matched_institution_id)
        if institution is None:
            raise ApplicationRejected(
                "organisation_introuvable",
                "L'institution notée n'existe plus.",
                status=404,
            )
        return institution
    matches = (
        Institution.query.filter(
            func.lower(Institution.name) == (row.establishment_name or "").lower()
        )
        .limit(2)
        .all()
    )
    if len(matches) == 1:
        raise ApplicationRejected(
            "organisation_existante",
            "Une institution correspond déjà. Notez-la explicitement avant d'approuver.",
            status=409,
        )
    if len(matches) > 1:
        raise ApplicationRejected(
            "organisation_ambigue",
            "Plusieurs institutions correspondent. Choisissez-en une explicitement.",
            status=409,
        )
    institution = Institution(
        name=(row.establishment_name or "")[:200],
        public_id=str(uuid.uuid4()),
        institution_type=row.establishment_category,
    )
    db.session.add(institution)
    db.session.flush()
    return institution


def _ensure_grant(
    user: User,
    portal: str,
    *,
    company_id: int | None = None,
    institution_id: int | None = None,
    status: str,
    internal_role: str | None,
) -> PortalGrant:
    query = PortalGrant.query.filter_by(user_id=user.id, portal=portal)
    if company_id is not None:
        query = query.filter_by(company_id=company_id)
    if institution_id is not None:
        query = query.filter_by(institution_id=institution_id)
    existing = query.one_or_none()
    if existing is not None:
        if existing.status == PortalGrantStatus.REVOKED.value:
            raise ApplicationRejected(
                "habilitation_revoquee",
                "Une habilitation révoquée ne peut pas être recréée par cette approbation.",
                status=409,
            )
        return existing
    grant = PortalGrant(
        user_id=user.id,
        portal=portal,
        company_id=company_id,
        institution_id=institution_id,
        internal_role=internal_role,
        status=status,
    )
    db.session.add(grant)
    db.session.flush()
    return grant


def _institution_grant(row: OrganizationApplication) -> PortalGrant | None:
    if row.user_id is None or row.matched_institution_id is None:
        return None
    return PortalGrant.query.filter_by(
        user_id=row.user_id,
        portal=PortalKind.INSTITUTION.value,
        institution_id=row.matched_institution_id,
    ).one_or_none()


def _grant_for(row: OrganizationApplication) -> PortalGrant | None:
    if row.kind == OrganizationApplicationKind.INSTITUTION.value:
        return _institution_grant(row)
    if row.user_id is None or row.matched_company_id is None:
        return None
    return PortalGrant.query.filter_by(
        user_id=row.user_id,
        portal=PortalKind.COMPANY.value,
        company_id=row.matched_company_id,
    ).one_or_none()
