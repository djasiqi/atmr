"""Instruction administrative des préinscriptions.

Aucune fonction de ce module ne crée de société, d'institution ou d'habilitation.
L'approbation qui ouvrirait un portail reste refusée.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from flask import current_app
from flask_mail import Message
from sqlalchemy.orm import joinedload

from ext import db, mail
from models import Company, Institution, User, UserRole
from models.enums import OrganizationApplicationKind, OrganizationApplicationStatus
from models.organization_application import (
    OrganizationApplication,
    OrganizationApplicationEvent,
)
from services.organization_application_gate import approval_enabled
from services.organization_applications import (
    SLA_NOTICE,
    ApplicationRejected,
    _persist,
)

logger = logging.getLogger(__name__)

_OPEN_FOR_REVIEW = {
    OrganizationApplicationStatus.RECEIVED.value,
    OrganizationApplicationStatus.IN_REVIEW.value,
    OrganizationApplicationStatus.NEEDS_INFO.value,
}
_KIND_LABELS = {
    OrganizationApplicationKind.COMPANY.value: "Entreprise",
    OrganizationApplicationKind.INSTITUTION.value: "Institution",
}
_STATUS_LABELS = {
    OrganizationApplicationStatus.EMAIL_UNVERIFIED.value: "Adresse e-mail à confirmer",
    OrganizationApplicationStatus.RECEIVED.value: "Dossier reçu",
    OrganizationApplicationStatus.IN_REVIEW.value: "En cours d'examen",
    OrganizationApplicationStatus.NEEDS_INFO.value: "Complément demandé",
    OrganizationApplicationStatus.APPROVED.value: "Demande approuvée",
    OrganizationApplicationStatus.REFUSED.value: "Demande refusée",
    OrganizationApplicationStatus.ABANDONED.value: "Demande abandonnée",
}
_EVENT_LABELS = {
    "created": "Demande enregistrée",
    "verification_sent": "Lien de confirmation envoyé",
    "email_verified": "Adresse e-mail confirmée",
    "received": "Dossier reçu",
    "review_started": "Examen pris en charge",
    "info_requested": "Complément demandé",
    "info_received": "Complément reçu",
    "review_resumed": "Examen repris",
    "refused": "Demande refusée",
    "approved": "Demande approuvée",
    "institution_invitation_sent": "Invitation du premier administrateur envoyée",
    "institution_admin_activated": "Premier administrateur activé",
    "grant_revoked": "Habilitation révoquée",
    "identity_confirmed": "Identité du représentant confirmée",
    "identity_rejected": "Identité du représentant non confirmée",
    "organization_confirmed": "Organisation confirmée",
    "organization_rejected": "Organisation non confirmée",
    "existing_org_noted": "Organisation existante notée",
    "observation": "Observation administrative",
    "notification_queued": "Notification mise en file",
    "notification_sent": "Notification remise au service de courriel",
    "notification_failed": "Notification non remise",
}
_APPROVAL_DISABLED = (
    "L'approbation qui crée une organisation ou une habilitation "
    "n'est pas disponible."
)


def list_applications(
    *,
    query: str = "",
    kind: str = "",
    status: str = "",
) -> dict[str, Any]:
    """Liste les dossiers. Un e-mail non confirmé n'est jamais un dossier complet."""
    rows = (
        OrganizationApplication.query.order_by(
            OrganizationApplication.submitted_at.desc().nullslast(),
            OrganizationApplication.created_at.desc(),
        )
        .all()
    )
    needle = query.strip().lower()
    items = []
    for row in rows:
        if kind and row.kind != kind:
            continue
        if status and row.status != status:
            continue
        if needle and needle not in _search_blob(row):
            continue
        items.append(_summary(row))
    return {
        "items": items,
        "approval_enabled": False,
        "sla_calendar": "pending",
    }


def application_detail(public_id: str) -> dict[str, Any]:
    row = _find(public_id)
    return _detail(row)


def search_organizations(needle: str) -> dict[str, list[dict[str, Any]]]:
    """Recherche en lecture seule. Aucune organisation n'est créée."""
    term = _like_term(needle)
    if not term:
        return {"companies": [], "institutions": []}
    companies = (
        Company.query.filter(Company.name.ilike(term, escape="\\"))
        .order_by(Company.name.asc())
        .limit(8)
        .all()
    )
    institutions = (
        Institution.query.filter(Institution.name.ilike(term, escape="\\"))
        .order_by(Institution.name.asc())
        .limit(8)
        .all()
    )
    return {
        "companies": [{"id": row.id, "name": row.name} for row in companies],
        "institutions": [{"id": row.id, "name": row.name} for row in institutions],
    }


def apply_decision(
    public_id: str,
    actor: User,
    action: str,
    note: str | None = None,
    *,
    confirmed: bool | None = None,
    company_id: int | None = None,
    institution_id: int | None = None,
) -> dict[str, Any]:
    """Applique une instruction. Rejoue sans doublon si l'état est déjà celui visé."""
    row = _locked(public_id)
    _reject_self_review(row, actor)
    logger.info(
        "instruction dossier=%s action=%s",
        row.public_id,
        action,
    )
    notification = None
    if action == "start_review":
        notification = _start_review(row, actor)
    elif action == "request_info":
        notification = _request_info(row, actor, note)
    elif action == "record_info":
        notification = _record_info(row, actor, note)
    elif action == "resume_review":
        notification = _resume_review(row, actor)
    elif action == "refuse":
        notification = _refuse(row, actor, note)
    elif action == "check_identity":
        _check_flag(row, actor, "identity", confirmed, note)
    elif action == "check_organization":
        _check_flag(row, actor, "organization", confirmed, note)
    elif action == "note_existing_organization":
        _note_existing(row, actor, company_id, institution_id, note)
    elif action == "observe":
        text = _required_note(note)
        _event(row, "observation", actor, text, internal=True)
    else:
        raise ApplicationRejected("action_inconnue", "Cette action n'existe pas.")
    _persist()
    db.session.expire(row, ["events"])
    detail = _detail(row)
    detail["notification"] = notification
    return detail


def refuse_activation(public_id: str, actor: User) -> None:
    """L'approbation effective reste fermée, même pour un administrateur."""
    row = _locked(public_id)
    _reject_self_review(row, actor)
    logger.info("approbation bloquee dossier=%s", row.public_id)
    raise ApplicationRejected("activation_bloquee", _APPROVAL_DISABLED, status=403)


def _start_review(row: OrganizationApplication, actor: User) -> dict[str, Any] | None:
    if row.status == OrganizationApplicationStatus.IN_REVIEW.value:
        return None
    _require_received(row)
    if row.status != OrganizationApplicationStatus.RECEIVED.value:
        raise ApplicationRejected(
            "transition_refusee",
            "Ce dossier ne peut pas être pris en charge depuis son état actuel.",
            status=409,
        )
    row.status = OrganizationApplicationStatus.IN_REVIEW.value
    _event(row, "review_started", actor, None, internal=False)
    return _notify(
        row,
        "review_started",
        "Votre demande est en cours d'examen. Aucun accès professionnel n'est ouvert.",
    )


def _request_info(row: OrganizationApplication, actor: User, note: str | None) -> dict[str, Any] | None:
    if row.status == OrganizationApplicationStatus.NEEDS_INFO.value:
        return None
    text = _required_note(note)
    if row.status != OrganizationApplicationStatus.IN_REVIEW.value:
        raise ApplicationRejected(
            "transition_refusee",
            "Le complément se demande pendant l'examen.",
            status=409,
        )
    row.status = OrganizationApplicationStatus.NEEDS_INFO.value
    _event(row, "info_requested", actor, text, internal=False)
    return _notify(
        row,
        "info_requested",
        "LIRIE a besoin d'un complément pour poursuivre l'examen :\n\n"
        f"{text}\n\nAucun accès professionnel n'est ouvert.",
    )


def _record_info(row: OrganizationApplication, actor: User, note: str | None) -> dict[str, Any] | None:
    text = _required_note(note)
    if row.status != OrganizationApplicationStatus.NEEDS_INFO.value:
        raise ApplicationRejected(
            "transition_refusee",
            "Un complément se reçoit seulement après qu'il a été demandé.",
            status=409,
        )
    if _last_event_type(row) == "info_received" and _last_note(row) == text:
        return None
    _event(row, "info_received", actor, text, internal=False)
    return _notify(
        row,
        "info_received",
        "Le complément annoncé a été enregistré. L'examen n'est pas encore repris.",
    )


def _resume_review(row: OrganizationApplication, actor: User) -> dict[str, Any] | None:
    if row.status == OrganizationApplicationStatus.IN_REVIEW.value:
        return None
    if row.status != OrganizationApplicationStatus.NEEDS_INFO.value:
        raise ApplicationRejected(
            "transition_refusee",
            "La reprise concerne un dossier en attente de complément.",
            status=409,
        )
    if not _complement_received(row):
        raise ApplicationRejected(
            "complement_absent",
            "Enregistrez d'abord la réception du complément.",
        )
    row.status = OrganizationApplicationStatus.IN_REVIEW.value
    _event(row, "review_resumed", actor, None, internal=False)
    return _notify(
        row,
        "review_resumed",
        "L'examen de votre demande a repris. Aucun accès professionnel n'est ouvert.",
    )


def _refuse(row: OrganizationApplication, actor: User, note: str | None) -> dict[str, Any] | None:
    if row.status == OrganizationApplicationStatus.REFUSED.value:
        return None
    if row.status == OrganizationApplicationStatus.EMAIL_UNVERIFIED.value:
        raise ApplicationRejected(
            "dossier_incomplet",
            "Ce dossier n'est pas complet : l'adresse e-mail n'est pas confirmée.",
        )
    text = _required_note(note)
    if row.status not in _OPEN_FOR_REVIEW:
        raise ApplicationRejected(
            "transition_refusee",
            "Ce dossier ne peut pas être refusé depuis son état actuel.",
            status=409,
        )
    row.status = OrganizationApplicationStatus.REFUSED.value
    _event(row, "refused", actor, text, internal=False)
    return _notify(
        row,
        "refused",
        "Votre demande n'est pas retenue :\n\n"
        f"{text}\n\nAucun accès professionnel n'a été ouvert.",
    )


def _check_flag(
    row: OrganizationApplication,
    actor: User,
    subject: str,
    confirmed: bool | None,
    note: str | None,
) -> None:
    if confirmed is None:
        raise ApplicationRejected("confirmation_requise", "Indiquez si la vérification est positive.")
    text = _required_note(note)
    prefix = "identity" if subject == "identity" else "organization"
    event_type = f"{prefix}_confirmed" if confirmed else f"{prefix}_rejected"
    _event(row, event_type, actor, text, internal=True)


def _note_existing(
    row: OrganizationApplication,
    actor: User,
    company_id: int | None,
    institution_id: int | None,
    note: str | None,
) -> None:
    if bool(company_id) == bool(institution_id):
        raise ApplicationRejected(
            "organisation_ambigue",
            "Notez une seule organisation existante.",
        )
    if row.kind == OrganizationApplicationKind.COMPANY.value:
        if not company_id:
            raise ApplicationRejected(
                "organisation_incompatible",
                "Ce dossier concerne une entreprise.",
            )
        company = db.session.get(Company, company_id)
        if company is None:
            raise ApplicationRejected(
                "organisation_introuvable",
                "Entreprise introuvable.",
                status=404,
            )
        row.matched_company_id = company.id
        label = company.name
    else:
        if not institution_id:
            raise ApplicationRejected(
                "organisation_incompatible",
                "Ce dossier concerne une institution.",
            )
        institution = db.session.get(Institution, institution_id)
        if institution is None:
            raise ApplicationRejected(
                "organisation_introuvable",
                "Institution introuvable.",
                status=404,
            )
        row.matched_institution_id = institution.id
        label = institution.name
    text = _optional_note(note)
    _event(
        row,
        "existing_org_noted",
        actor,
        text or f"Correspondance notée : {label}",
        internal=True,
    )


def _notify(row: OrganizationApplication, template: str, body: str) -> dict[str, Any]:
    _event(row, "notification_queued", None, template, internal=True)
    if current_app.config.get("TESTING"):
        current_app.config.setdefault("_ORG_APP_NOTICES", []).append(
            {
                "email": row.applicant_email,
                "template": template,
                "body": body,
            }
        )
        return {"state": "queued", "delivered": False, "template": template}
    return _hand_to_mailer(row, template, body)


def _hand_to_mailer(row: OrganizationApplication, template: str, body: str) -> dict[str, Any]:
    if current_app.config.get("MAIL_SUPPRESS_SEND", True):
        return {"state": "queued", "delivered": False, "template": template}
    sender = (
        current_app.config.get("MAIL_DEFAULT_SENDER")
        or current_app.config.get("MAIL_USERNAME")
        or "noreply@lirie.ch"
    )
    message = Message(
        subject="Votre demande LIRIE",
        sender=sender,
        recipients=[row.applicant_email],
        body=f"{body}\n\n{SLA_NOTICE}",
    )
    try:
        mail.send(message)
    except Exception:
        logger.exception("Notification de préinscription impossible")
        _event(row, "notification_failed", None, template, internal=True)
        return {"state": "failed", "delivered": False, "template": template}
    _event(row, "notification_sent", None, template, internal=True)
    return {"state": "handed_to_mailer", "delivered": False, "template": template}


def _require_received(row: OrganizationApplication) -> None:
    if row.status == OrganizationApplicationStatus.EMAIL_UNVERIFIED.value:
        raise ApplicationRejected(
            "dossier_incomplet",
            "Ce dossier n'est pas complet : l'adresse e-mail n'est pas confirmée.",
        )


def _complement_received(row: OrganizationApplication) -> bool:
    requested_at = None
    received = False
    for event in _ordered(row):
        if event.event_type == "info_requested":
            requested_at = event.created_at
            received = False
        elif event.event_type == "info_received" and requested_at is not None:
            received = True
    return received


def _reject_self_review(row: OrganizationApplication, actor: User) -> None:
    same_user = row.user_id is not None and row.user_id == actor.id
    same_email = (
        actor.email
        and row.applicant_email
        and actor.email.lower() == row.applicant_email.lower()
    )
    if same_user or same_email:
        raise ApplicationRejected(
            "auto_decision_refusee",
            "Une personne ne peut pas instruire sa propre demande.",
            status=403,
        )


def _event(
    row: OrganizationApplication,
    event_type: str,
    actor: User | None,
    note: str | None,
    *,
    internal: bool,
) -> None:
    db.session.add(
        OrganizationApplicationEvent(
            application=row,
            event_type=event_type,
            visibility="internal" if internal else "public",
            actor_user_id=None if actor is None else actor.id,
            note=note,
        )
    )


def _summary(row: OrganizationApplication) -> dict[str, Any]:
    incomplete = row.status == OrganizationApplicationStatus.EMAIL_UNVERIFIED.value
    return {
        "public_id": row.public_id,
        "kind": row.kind,
        "kind_label": _KIND_LABELS.get(row.kind, row.kind),
        "status": row.status,
        "status_label": _STATUS_LABELS.get(row.status, row.status),
        "organization_name": row.organization_label(),
        "applicant_name": f"{row.applicant_first_name} {row.applicant_last_name}".strip(),
        "applicant_email": row.applicant_email,
        "city": row.city,
        "submitted_at": _iso(row.submitted_at),
        "created_at": _iso(row.created_at),
        "incomplete": incomplete,
        "actionable": row.status in _OPEN_FOR_REVIEW,
        "queue_label": _queue_label(row.status),
    }


def _approval_state() -> dict[str, Any]:
    if approval_enabled():
        return {
            "enabled": True,
            "reason": (
                "L'approbation prépare l'organisation et l'habilitation. "
                "Elle n'est proposée que si le verrou de ce processus est ouvert."
            ),
        }
    return {"enabled": False, "reason": _APPROVAL_DISABLED}


def _detail(row: OrganizationApplication) -> dict[str, Any]:
    payload = _summary(row)
    payload.update(
        {
            "applicant_first_name": row.applicant_first_name,
            "applicant_last_name": row.applicant_last_name,
            "applicant_function": row.applicant_function,
            "applicant_phone": row.applicant_phone,
            "street": row.street,
            "postal_code": row.postal_code,
            "country": row.country,
            "ide_number": row.ide_number,
            "establishment_category": row.establishment_category,
            "applicant_review_note": row.review_note,
            "email_verified": row.email_verified_at is not None,
            "matched_company_id": row.matched_company_id,
            "matched_institution_id": row.matched_institution_id,
            "documents": [],
            "documents_note": "Aucun justificatif n'est accepté pour le moment.",
            "approval": _approval_state(),
            "sla": _sla(row),
            "checks": _checks(row),
            "events": [_event_view(event) for event in _ordered(row)],
        }
    )
    return payload


def _sla(row: OrganizationApplication) -> dict[str, Any]:
    suspensions = []
    opened = None
    for event in _ordered(row):
        if event.event_type == "info_requested":
            opened = event.created_at
        elif event.event_type == "info_received" and opened is not None:
            suspensions.append({"from": _iso(opened), "until": _iso(event.created_at)})
            opened = None
    if opened is not None:
        suspensions.append({"from": _iso(opened), "until": None})
    return {
        "notice": SLA_NOTICE,
        "calendar": "pending",
        "due_at": None,
        "suspensions": suspensions,
    }


def _checks(row: OrganizationApplication) -> dict[str, Any]:
    return {
        "identity": _latest_flag(row, "identity_confirmed", "identity_rejected"),
        "organization": _latest_flag(row, "organization_confirmed", "organization_rejected"),
    }


def _latest_flag(row: OrganizationApplication, yes: str, no: str) -> dict[str, Any]:
    found = None
    for event in _ordered(row):
        if event.event_type in {yes, no}:
            found = event
    if found is None:
        return {"confirmed": None, "at": None, "note": None}
    return {
        "confirmed": found.event_type == yes,
        "at": _iso(found.created_at),
        "note": found.note,
    }


def _event_view(event: OrganizationApplicationEvent) -> dict[str, Any]:
    actor = None
    if event.actor_user_id is not None:
        user = db.session.get(User, event.actor_user_id)
        actor = None if user is None else user.username
    return {
        "type": event.event_type,
        "label": _EVENT_LABELS.get(event.event_type, event.event_type),
        "visibility": event.visibility,
        "note": event.note,
        "actor": actor,
        "at": _iso(event.created_at),
    }


def _queue_label(status: str) -> str:
    if status == OrganizationApplicationStatus.EMAIL_UNVERIFIED.value:
        return "Incomplet"
    if status == OrganizationApplicationStatus.RECEIVED.value:
        return "À traiter"
    if status == OrganizationApplicationStatus.NEEDS_INFO.value:
        return "Complément en attente"
    if status == OrganizationApplicationStatus.IN_REVIEW.value:
        return "En examen"
    return "Clos"


def _search_blob(row: OrganizationApplication) -> str:
    return " ".join(
        part.lower()
        for part in (
            row.organization_label(),
            row.applicant_first_name,
            row.applicant_last_name,
            row.applicant_email,
            row.city,
            row.ide_number or "",
        )
    )


def _find(public_id: str) -> OrganizationApplication:
    row = (
        OrganizationApplication.query.options(joinedload(OrganizationApplication.events))
        .filter_by(public_id=public_id)
        .one_or_none()
    )
    if row is None:
        raise ApplicationRejected("dossier_introuvable", "Dossier introuvable.", status=404)
    return row


def _locked(public_id: str) -> OrganizationApplication:
    row = (
        OrganizationApplication.query.filter_by(public_id=public_id)
        .with_for_update()
        .one_or_none()
    )
    if row is None:
        raise ApplicationRejected("dossier_introuvable", "Dossier introuvable.", status=404)
    return row


def _ordered(row: OrganizationApplication) -> list[OrganizationApplicationEvent]:
    return sorted(
        row.events,
        key=lambda item: (
            item.created_at or datetime.min.replace(tzinfo=UTC),
            item.id or 0,
        ),
    )


def _last_event_type(row: OrganizationApplication) -> str | None:
    events = _ordered(row)
    return None if not events else events[-1].event_type


def _last_note(row: OrganizationApplication) -> str | None:
    events = _ordered(row)
    return None if not events else events[-1].note


def _required_note(note: str | None) -> str:
    text = str(note or "").strip()
    if len(text) < 10:
        raise ApplicationRejected(
            "motif_requis",
            "Indiquez un motif d'au moins 10 caractères.",
        )
    if len(text) > 2000:
        raise ApplicationRejected("motif_trop_long", "Le motif est trop long.")
    return text


def _optional_note(note: str | None) -> str | None:
    text = str(note or "").strip()
    if len(text) > 2000:
        raise ApplicationRejected("motif_trop_long", "Le motif est trop long.")
    return text or None


def _like_term(needle: str) -> str:
    cleaned = (
        needle.strip()
        .replace("\\", "")
        .replace("%", "")
        .replace("_", "")
    )
    if len(cleaned) < 2:
        return ""
    return f"%{cleaned}%"


def _iso(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def load_actor(public_id: str) -> User:
    user = User.query.filter_by(public_id=str(public_id)).one_or_none()
    if user is None or user.role != UserRole.ADMIN:
        raise ApplicationRejected(
            "admin_requis",
            "Accès réservé à l'administration LIRIE.",
            status=403,
        )
    return user
