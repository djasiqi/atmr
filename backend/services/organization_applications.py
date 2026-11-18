"""Dépôt et suivi d'une préinscription. Aucun droit professionnel n'est créé."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from flask import current_app
from flask_mail import Message
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError

from ext import db, mail
from models import Company, Institution, User
from models.enums import OrganizationApplicationKind, OrganizationApplicationStatus
from models.organization_application import (
    OrganizationApplication,
    OrganizationApplicationEvent,
    OrganizationApplicationToken,
)

logger = logging.getLogger(__name__)

GENERIC_ACCEPTANCE = (
    "Si cette adresse peut recevoir un message, un lien de confirmation "
    "vient d'être envoyé. Il reste valable 24 heures."
)
SLA_NOTICE = (
    "Votre demande sera examinée dans un délai maximal de 24 heures "
    "administratives cumulées, à compter de la réception de votre dossier complet."
)
INVALID_LINK = (
    "Ce lien n'est plus valable. Demandez un nouvel envoi depuis le formulaire."
)
VERIFY_TTL = timedelta(hours=24)
TRACK_TTL = timedelta(days=14)
OPEN_STATUSES = (
    OrganizationApplicationStatus.EMAIL_UNVERIFIED.value,
    OrganizationApplicationStatus.RECEIVED.value,
    OrganizationApplicationStatus.IN_REVIEW.value,
    OrganizationApplicationStatus.NEEDS_INFO.value,
)
FORBIDDEN_KEYS = frozenset(
    {
        "role",
        "user_role",
        "portal",
        "portal_grant",
        "company_id",
        "institution_id",
        "status",
        "internal_role",
    }
)
INSTITUTION_CATEGORIES = frozenset(
    {"hopital", "clinique", "ems", "imad", "curatelle", "autre"}
)
_IDE_RE = re.compile(
    r"^CHE[- ]?\d{3}\.\d{3}\.\d{3}(\s*TVA)?$|^CHE[- ]?\d{9}(\s*TVA)?$",
    flags=re.IGNORECASE,
)


def _is_valid_email(value: str) -> bool:
    """Validation e-mail bornée (évite ReDoS CodeQL sur quantifiers imbriqués)."""
    if len(value) > 255 or value.count("@") != 1:
        return False
    local, domain = value.split("@", 1)
    if not local or not domain or " " in value or "\t" in value:
        return False
    return "." in domain and not domain.startswith(".") and not domain.endswith(".")


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
}
_STATUS_LABELS = {
    OrganizationApplicationStatus.EMAIL_UNVERIFIED.value: "Adresse e-mail à confirmer",
    OrganizationApplicationStatus.RECEIVED.value: "Dossier reçu",
    OrganizationApplicationStatus.IN_REVIEW.value: "En cours d'examen",
    OrganizationApplicationStatus.NEEDS_INFO.value: "Informations complémentaires demandées",
    OrganizationApplicationStatus.APPROVED.value: "Demande approuvée",
    OrganizationApplicationStatus.REFUSED.value: "Demande refusée",
    OrganizationApplicationStatus.ABANDONED.value: "Demande abandonnée",
}


class ApplicationRejected(Exception):
    """Refus explicite, sans création de dossier ni de droit."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def submit_application(payload: dict[str, Any]) -> dict[str, str]:
    """Enregistre un dossier complet ou renvoie le même accusé."""
    if not isinstance(payload, dict):
        raise ApplicationRejected("payload_invalide", "Le dossier est incomplet.")
    forbidden = FORBIDDEN_KEYS.intersection(payload)
    if forbidden:
        raise ApplicationRejected(
            "droit_refuse",
            "Cette demande ne peut pas attribuer de droit professionnel.",
        )
    if str(payload.get("website") or "").strip():
        return {"message": GENERIC_ACCEPTANCE}
    fields = _validated_fields(payload)
    try:
        existing = _open_application(fields["applicant_email"], fields["kind"])
        if existing is None:
            application = _build_application(fields)
            db.session.add(application)
            db.session.flush()
            _add_event(application, "created")
        elif existing.status == OrganizationApplicationStatus.EMAIL_UNVERIFIED.value:
            application = existing
            _copy_fields(application, fields)
        else:
            return {"message": GENERIC_ACCEPTANCE}
        raw_token = _rotate_verification_token(application)
        _add_event(application, "verification_sent")
        _persist()
    except IntegrityError:
        db.session.rollback()
        return {"message": GENERIC_ACCEPTANCE}
    deliver_verification_link(application.applicant_email, raw_token)
    return {"message": GENERIC_ACCEPTANCE}


def verify_email_token(raw_token: str) -> dict[str, Any]:
    """Passe un dossier complet de email_unverified à received."""
    token = _usable_token(raw_token, "email_verify")
    if token is None:
        raise ApplicationRejected("lien_invalide", INVALID_LINK)
    now = datetime.now(UTC)
    token.used_at = now
    application = token.application
    if application.status == OrganizationApplicationStatus.EMAIL_UNVERIFIED.value:
        application.status = OrganizationApplicationStatus.RECEIVED.value
        application.email_verified_at = now
        application.submitted_at = now
        _add_event(application, "email_verified")
        _add_event(application, "received")
    tracking = _issue_token(application, "tracking", TRACK_TTL)
    _persist()
    view = public_view(application)
    view["tracking_token"] = tracking
    return view


def read_with_tracking_token(raw_token: str) -> dict[str, Any]:
    """Retourne uniquement le dossier lié à ce jeton de suivi."""
    token = _usable_token(raw_token, "tracking")
    if token is None:
        raise ApplicationRejected("suivi_refuse", "Dossier introuvable.", status=401)
    return public_view(token.application)


def _public_events(application: OrganizationApplication) -> list:
    visible = [
        event
        for event in application.events
        if getattr(event, "visibility", "public") != "internal"
    ]
    return sorted(
        visible,
        key=lambda item: (
            item.created_at or datetime.min.replace(tzinfo=UTC),
            item.id or 0,
        ),
    )


def public_view(application: OrganizationApplication) -> dict[str, Any]:
    hint = (
        application.matched_company_id is not None
        or application.matched_institution_id is not None
    )
    return {
        "kind": application.kind,
        "kind_label": (
            "Entreprise de transport"
            if application.kind == OrganizationApplicationKind.COMPANY.value
            else "Institution"
        ),
        "status": application.status,
        "status_label": _STATUS_LABELS.get(application.status, application.status),
        "organization_name": application.organization_label(),
        "applicant_name": (
            f"{application.applicant_first_name} {application.applicant_last_name}"
        ).strip(),
        "applicant_email": application.applicant_email,
        "sla_notice": SLA_NOTICE,
        "existing_organization_hint": hint,
        "email_verified": application.email_verified_at is not None,
        "events": [
            {
                "type": event.event_type,
                "label": _EVENT_LABELS.get(event.event_type, event.event_type),
                "note": event.note,
                "at": event.created_at.isoformat() if event.created_at else None,
            }
            for event in _public_events(application)
        ],
    }


def deliver_verification_link(email: str, raw_token: str) -> None:
    """Envoie le lien. Le secret n'est pas journalisé."""
    if current_app.config.get("TESTING"):
        box = current_app.config.setdefault("_ORG_APP_LINKS", [])
        box.append({"email": email, "token": raw_token})
        return
    link = f"{_frontend_base()}/inscription/organisation/verifier?token={raw_token}"
    sender = (
        current_app.config.get("MAIL_DEFAULT_SENDER")
        or current_app.config.get("MAIL_USERNAME")
        or "noreply@lirie.ch"
    )
    message = Message(
        subject="Confirmez votre demande LIRIE",
        sender=sender,
        recipients=[email],
        body=(
            "Confirmez votre adresse pour que LIRIE enregistre votre demande.\n\n"
            f"{link}\n\n"
            "Ce lien expire dans 24 heures. "
            "Aucun accès professionnel n'est ouvert à cette étape.\n\n"
            f"{SLA_NOTICE}"
        ),
    )
    try:
        mail.send(message)
    except Exception:
        logger.warning("Envoi du lien de préinscription impossible")


def _validated_fields(payload: dict[str, Any]) -> dict[str, Any]:
    kind = str(payload.get("kind") or "").strip()
    if kind not in {
        OrganizationApplicationKind.COMPANY.value,
        OrganizationApplicationKind.INSTITUTION.value,
    }:
        raise ApplicationRejected(
            "kind_invalide", "Choisissez une entreprise ou une institution."
        )
    email = str(payload.get("applicant_email") or "").strip().lower()
    if not _is_valid_email(email):
        raise ApplicationRejected(
            "email_invalide", "Indiquez une adresse e-mail professionnelle valide."
        )
    phone = re.sub(r"[^\d+]", "", str(payload.get("applicant_phone") or ""))
    if len(phone) < 8 or len(phone) > 30:
        raise ApplicationRejected(
            "telephone_invalide", "Indiquez un téléphone professionnel valide."
        )
    first_name = _required_text(payload, "applicant_first_name", "le prénom", 80)
    last_name = _required_text(payload, "applicant_last_name", "le nom", 80)
    function = _required_text(payload, "applicant_function", "la fonction", 80)
    street = _required_text(payload, "street", "la rue", 255)
    city = _required_text(payload, "city", "la localité", 100)
    postal_code = str(payload.get("postal_code") or "").strip()
    if not re.fullmatch(r"\d{4}", postal_code):
        raise ApplicationRejected(
            "npa_invalide", "Indiquez un code postal suisse à 4 chiffres."
        )
    if payload.get("privacy_consent") is not True:
        raise ApplicationRejected(
            "consentement_requis",
            "Le consentement au traitement du dossier est requis.",
        )
    note = str(payload.get("review_note") or "").strip()
    if len(note) > 500:
        raise ApplicationRejected(
            "note_trop_longue", "La précision dépasse 500 caractères."
        )
    fields: dict[str, Any] = {
        "kind": kind,
        "applicant_email": email,
        "applicant_phone": phone,
        "applicant_first_name": first_name,
        "applicant_last_name": last_name,
        "applicant_function": function,
        "street": street,
        "city": city,
        "postal_code": postal_code,
        "review_note": note or None,
        "legal_name": None,
        "ide_number": None,
        "establishment_name": None,
        "establishment_category": None,
    }
    if kind == OrganizationApplicationKind.COMPANY.value:
        fields["legal_name"] = _required_text(
            payload, "legal_name", "la raison sociale", 200
        )
        raw_ide = str(payload.get("ide_number") or "").strip()
        fields["ide_number"] = _normalize_ide(raw_ide) if raw_ide else None
    else:
        fields["establishment_name"] = _required_text(
            payload, "establishment_name", "le nom de l'établissement", 200
        )
        category = str(payload.get("establishment_category") or "").strip()
        if category not in INSTITUTION_CATEGORIES:
            raise ApplicationRejected(
                "categorie_invalide",
                "Choisissez la catégorie de l'établissement.",
            )
        fields["establishment_category"] = category
    return fields


def _build_application(fields: dict[str, Any]) -> OrganizationApplication:
    email = fields["applicant_email"]
    users = User.query.filter(func.lower(User.email) == email).limit(2).all()
    user_id = int(users[0].id) if len(users) == 1 else None
    company_id, institution_id = _match_organization(fields)
    return OrganizationApplication(
        public_id=str(uuid.uuid4()),
        kind=fields["kind"],
        status=OrganizationApplicationStatus.EMAIL_UNVERIFIED.value,
        applicant_first_name=fields["applicant_first_name"],
        applicant_last_name=fields["applicant_last_name"],
        applicant_function=fields["applicant_function"],
        applicant_email=email,
        applicant_phone=fields["applicant_phone"],
        legal_name=fields["legal_name"],
        ide_number=fields["ide_number"],
        establishment_name=fields["establishment_name"],
        establishment_category=fields["establishment_category"],
        street=fields["street"],
        postal_code=fields["postal_code"],
        city=fields["city"],
        country="CH",
        review_note=fields["review_note"],
        matched_company_id=company_id,
        matched_institution_id=institution_id,
        user_id=user_id,
    )


def _match_organization(fields: dict[str, Any]) -> tuple[int | None, int | None]:
    if fields["kind"] == OrganizationApplicationKind.COMPANY.value:
        filters = [func.lower(Company.name) == fields["legal_name"].lower()]
        if fields["ide_number"]:
            filters.append(Company.uid_ide == fields["ide_number"])
        rows = Company.query.filter(or_(*filters)).limit(2).all()
        if len(rows) == 1:
            return int(rows[0].id), None
        return None, None
    rows = (
        Institution.query.filter(
            func.lower(Institution.name) == fields["establishment_name"].lower()
        )
        .limit(2)
        .all()
    )
    if len(rows) == 1:
        return None, int(rows[0].id)
    return None, None


def _open_application(email: str, kind: str) -> OrganizationApplication | None:
    return (
        OrganizationApplication.query.filter(
            OrganizationApplication.applicant_email == email,
            OrganizationApplication.kind == kind,
            OrganizationApplication.status.in_(OPEN_STATUSES),
        )
        .order_by(OrganizationApplication.id.desc())
        .first()
    )


def _copy_fields(application: OrganizationApplication, fields: dict[str, Any]) -> None:
    application.applicant_first_name = fields["applicant_first_name"]
    application.applicant_last_name = fields["applicant_last_name"]
    application.applicant_function = fields["applicant_function"]
    application.applicant_phone = fields["applicant_phone"]
    application.legal_name = fields["legal_name"]
    application.ide_number = fields["ide_number"]
    application.establishment_name = fields["establishment_name"]
    application.establishment_category = fields["establishment_category"]
    application.street = fields["street"]
    application.postal_code = fields["postal_code"]
    application.city = fields["city"]
    application.review_note = fields["review_note"]
    company_id, institution_id = _match_organization(fields)
    application.matched_company_id = company_id
    application.matched_institution_id = institution_id


def _rotate_verification_token(application: OrganizationApplication) -> str:
    now = datetime.now(UTC)
    for token in list(application.tokens):
        if token.purpose == "email_verify" and token.used_at is None:
            token.used_at = now
    return _issue_token(application, "email_verify", VERIFY_TTL)


def _issue_token(
    application: OrganizationApplication, purpose: str, ttl: timedelta
) -> str:
    raw = secrets.token_urlsafe(32)
    db.session.add(
        OrganizationApplicationToken(
            application=application,
            token_hash=_hash_token(raw),
            purpose=purpose,
            expires_at=datetime.now(UTC) + ttl,
        )
    )
    return raw


def _usable_token(raw_token: str, purpose: str) -> OrganizationApplicationToken | None:
    if not raw_token or len(raw_token) > 200:
        return None
    token = OrganizationApplicationToken.query.filter_by(
        token_hash=_hash_token(raw_token),
        purpose=purpose,
    ).one_or_none()
    if token is None or token.used_at is not None:
        return None
    expires_at = token.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at <= datetime.now(UTC):
        return None
    return token


def _add_event(application: OrganizationApplication, event_type: str) -> None:
    db.session.add(
        OrganizationApplicationEvent(
            application=application,
            event_type=event_type,
        )
    )


def _persist() -> None:
    db.session.flush()
    if not current_app.config.get("TESTING"):
        db.session.commit()


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _required_text(payload: dict[str, Any], key: str, label: str, limit: int) -> str:
    value = str(payload.get(key) or "").strip()
    if not value:
        raise ApplicationRejected("champ_manquant", f"Indiquez {label}.")
    if len(value) > limit:
        raise ApplicationRejected(
            "champ_trop_long", f"Le champ « {label} » est trop long."
        )
    return value


def _normalize_ide(value: str) -> str:
    if not _IDE_RE.match(value.strip()):
        raise ApplicationRejected(
            "ide_invalide",
            "L'IDE suisse doit ressembler à CHE-123.456.789.",
        )
    digits = re.sub(r"\D", "", value)[:9]
    normalized = f"CHE-{digits[0:3]}.{digits[3:6]}.{digits[6:9]}"
    if "TVA" in value.upper():
        normalized += " TVA"
    return normalized


def _frontend_base() -> str:
    environment = str(current_app.config.get("ENVIRONMENT", "")).strip().lower()
    default = (
        "http://localhost:3000"
        if environment in {"development", "testing", ""}
        else "https://www.lirie.ch"
    )
    return (os.getenv("FRONTEND_URL") or default).rstrip("/")
