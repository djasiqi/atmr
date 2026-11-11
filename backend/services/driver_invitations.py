"""Invitations chauffeur envoyées par une entreprise approuvée.

Le mot de passe n'est jamais choisi par l'entreprise. Une adresse e-mail
identique ne rattache pas un compte : le destinataire doit s'authentifier
ou créer son accès, puis accepter explicitement cette entreprise.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from flask import current_app
from flask_mail import Message
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from ext import db, mail
from models import Company, Driver, PortalGrant, User, UserRole
from models.driver_invitation import DriverInvitation
from models.enums import DriverType, PortalGrantStatus, PortalKind
from security.password_policy import PasswordPolicyError, PasswordPolicyService
from security.portal_scope import enforcement_enabled, portal_company_allowed
from services.driver_invitation_gate import driver_invitation_activation_enabled

logger = logging.getLogger(__name__)

INVITE_TTL = timedelta(days=7)
MAX_RESENDS = 5
MAX_ATTEMPTS = 8
CONTROLLED_DOMAINS = frozenset({"example.test", "example.invalid"})


def _is_valid_email(value: str) -> bool:
    """Validation e-mail bornée (évite ReDoS CodeQL sur quantifiers imbriqués)."""
    if len(value) > 255 or value.count("@") != 1:
        return False
    local, domain = value.split("@", 1)
    if not local or not domain or " " in value or "\t" in value:
        return False
    return "." in domain and not domain.startswith(".") and not domain.endswith(".")


_FORBIDDEN_CREATE = frozenset(
    {
        "password",
        "role",
        "user_role",
        "portal",
        "portal_grant",
        "company_id",
        "status",
        "driver_id",
        "user_id",
        "internal_role",
    }
)
_FORBIDDEN_ACCEPT = frozenset(
    {
        "role",
        "user_role",
        "portal",
        "portal_grant",
        "company_id",
        "status",
        "driver_id",
        "user_id",
        "internal_role",
    }
)
_STATUS_LABELS = {
    "pending": "En attente",
    "accepted": "Acceptée",
    "cancelled": "Annulée",
    "expired": "Expirée",
    "revoked": "Rattachement retiré",
}


class InvitationRejected(Exception):
    """Refus explicite, sans rattachement ni habilitation."""

    def __init__(self, code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def create_invitation(
    company: Company, actor: User, payload: dict[str, Any]
) -> dict[str, Any]:
    """Enregistre une invitation pour l'entreprise de l'acteur."""
    _guard(company, actor)
    _reject_forbidden(payload, _FORBIDDEN_CREATE)
    email = _email(payload.get("email"))
    first_name = _name(payload.get("first_name"), "le prénom")
    last_name = _name(payload.get("last_name"), "le nom")
    existing = DriverInvitation.query.filter_by(
        company_id=company.id,
        email=email,
        status="pending",
    ).one_or_none()
    if existing is not None:
        raise InvitationRejected(
            "invitation_en_cours",
            "Une invitation est déjà en attente pour cette adresse. Renvoyez-la.",
            status=409,
        )
    raw = _new_secret()
    row = DriverInvitation(
        public_id=str(uuid.uuid4()),
        company_id=int(company.id),
        invited_by_user_id=int(actor.id),
        email=email,
        first_name=first_name,
        last_name=last_name,
        token_hash=_hash_token(raw),
        status="pending",
        expires_at=datetime.now(UTC) + INVITE_TTL,
    )
    db.session.add(row)
    try:
        with db.session.begin_nested():
            _persist()
    except IntegrityError as error:
        raise InvitationRejected(
            "invitation_en_cours",
            "Une invitation est déjà en cours pour cette adresse.",
            status=409,
        ) from error
    notice = _notify(row, raw, "driver_invitation")
    logger.info("invitation_chauffeur_creee public_id=%s", row.public_id)
    payload_out = _company_item(row)
    payload_out["notification"] = notice
    return payload_out


def list_invitations(company: Company, actor: User) -> dict[str, Any]:
    """Liste les invitations de cette entreprise, sans le secret."""
    _guard(company, actor)
    rows = (
        DriverInvitation.query.filter_by(company_id=int(company.id))
        .order_by(DriverInvitation.created_at.desc())
        .all()
    )
    changed = False
    for row in rows:
        if _mark_expired(row):
            changed = True
    if changed:
        _persist()
    return {
        "invitations": [_company_item(row) for row in rows],
        "activation": _activation_state(),
    }


def resend_invitation(company: Company, actor: User, public_id: str) -> dict[str, Any]:
    """Remplace le lien. L'ancien jeton ne fonctionne plus."""
    _guard(company, actor)
    row = _owned(company, public_id, lock=True)
    if row.status == "cancelled":
        raise InvitationRejected(
            "invitation_annulee",
            "Cette invitation est annulée.",
        )
    if row.status in {"accepted", "revoked"}:
        raise InvitationRejected(
            "invitation_closee",
            "Cette invitation n'est plus en attente.",
            status=409,
        )
    if row.resend_count >= MAX_RESENDS:
        raise InvitationRejected(
            "renvoi_limite",
            "Cette invitation a déjà été renvoyée plusieurs fois.",
            status=429,
        )
    raw = _new_secret()
    row.token_hash = _hash_token(raw)
    row.status = "pending"
    row.expires_at = datetime.now(UTC) + INVITE_TTL
    row.resend_count = int(row.resend_count) + 1
    row.attempt_count = 0
    _persist()
    notice = _notify(row, raw, "driver_invitation_resent")
    logger.info("invitation_chauffeur_renvoyee public_id=%s", row.public_id)
    payload = _company_item(row)
    payload["notification"] = notice
    return payload


def cancel_invitation(company: Company, actor: User, public_id: str) -> dict[str, Any]:
    """Annule une invitation encore en attente. Le lien ne peut plus être accepté."""
    _guard(company, actor)
    row = _owned(company, public_id, lock=True)
    if row.status == "cancelled":
        return _company_item(row)
    if row.status != "pending":
        raise InvitationRejected(
            "invitation_closee",
            "Seule une invitation en attente peut être annulée.",
            status=409,
        )
    row.status = "cancelled"
    row.cancelled_at = datetime.now(UTC)
    _persist()
    logger.info("invitation_chauffeur_annulee public_id=%s", row.public_id)
    return _company_item(row)


def revoke_attachment(company: Company, actor: User, public_id: str) -> dict[str, Any]:
    """Retire seulement le chauffeur créé par cette invitation."""
    _guard(company, actor)
    row = _owned(company, public_id, lock=True)
    if row.status == "revoked":
        return {
            "status": "revoked",
            "driver_removed": False,
            "public_id": row.public_id,
        }
    if row.status != "accepted":
        raise InvitationRejected(
            "rattachement_absent",
            "Cette invitation n'a pas de rattachement à retirer.",
            status=409,
        )
    removed = _remove_created_driver(row)
    row.status = "revoked"
    row.revoked_at = datetime.now(UTC)
    row.created_driver_id = None
    _persist()
    logger.info("invitation_chauffeur_revoquee public_id=%s", row.public_id)
    return {
        "status": "revoked",
        "driver_removed": removed,
        "public_id": row.public_id,
    }


def open_invitation(raw_token: str) -> dict[str, Any]:
    """Présente l'entreprise invitante sans consommer le lien."""
    row = _by_token(raw_token, lock=False)
    if row is None:
        raise InvitationRejected(
            "lien_invalide",
            "Ce lien n'est pas valable.",
        )
    if _mark_expired(row):
        _persist()
    return _public_view(row)


def accept_invitation(
    payload: dict[str, Any], authenticated_user_id: int | None
) -> dict[str, Any]:
    """Accepte le rattachement à l'entreprise du lien, si le verrou est ouvert."""
    if not isinstance(payload, dict):
        raise InvitationRejected("payload_invalide", "La demande est incomplète.")
    _reject_forbidden(payload, _FORBIDDEN_ACCEPT)
    if payload.get("accept") is not True:
        raise InvitationRejected(
            "acceptation_requise",
            "Le rattachement doit être accepté explicitement.",
        )
    row = _by_token(str(payload.get("token") or ""), lock=True)
    if row is None:
        raise InvitationRejected("lien_invalide", "Ce lien n'est pas valable.")
    if _mark_expired(row):
        _persist()
        raise InvitationRejected(
            "lien_expire",
            "Ce lien a expiré. Demandez un nouvel envoi à votre entreprise.",
        )
    if row.status == "cancelled":
        raise InvitationRejected(
            "invitation_annulee",
            "Cette invitation a été annulée.",
        )
    if row.status == "revoked":
        raise InvitationRejected(
            "invitation_revoquee",
            "Ce rattachement a été retiré.",
        )
    if row.status == "accepted":
        return {
            "status": "accepted",
            "already_accepted": True,
            "notification": None,
            "activation": _activation_state(),
            "company_name": _company_name(row),
        }
    if row.status != "pending":
        raise InvitationRejected("lien_invalide", "Ce lien n'est pas valable.")
    company = db.session.get(Company, int(row.company_id))
    if company is None or not bool(company.is_approved):
        raise InvitationRejected(
            "entreprise_non_approuvee",
            "L'entreprise invitante n'est pas approuvée.",
            status=403,
        )
    stated_email = str(payload.get("email") or "").strip().lower()
    if stated_email and stated_email != row.email:
        raise InvitationRejected(
            "destinataire_different",
            "Ce lien ne correspond pas à l'adresse indiquée.",
            status=403,
        )
    if not driver_invitation_activation_enabled():
        raise InvitationRejected(
            "activation_bloquee",
            "L'activation des chauffeurs n'est pas ouverte. L'invitation reste en attente.",
            status=403,
        )
    account = _account_for(row.email)
    if account is None:
        if authenticated_user_id is not None:
            raise InvitationRejected(
                "destinataire_different",
                "Le compte connecté n'est pas le destinataire de cette invitation.",
                status=403,
            )
        _require_new_password(row, payload)
    else:
        _require_same_account(account, authenticated_user_id)
        _refuse_other_company(account, int(row.company_id))
    try:
        with db.session.begin_nested():
            user, created = _attach(row, account, payload)
            if created is not None:
                _ensure_driver_grant(user, int(row.company_id))
                if not portal_company_allowed(
                    user, PortalKind.DRIVER.value, int(row.company_id)
                ):
                    raise InvitationRejected(
                        "acces_non_ouvert",
                        "L'habilitation chauffeur n'ouvre pas le portail. "
                        "L'acceptation est annulée.",
                        status=409,
                    )
            row.status = "accepted"
            row.accepted_at = datetime.now(UTC)
            row.accepted_user_id = int(user.id)
            row.created_driver_id = int(created.id) if created is not None else None
            _persist()
    except InvitationRejected:
        db.session.expire_all()
        raise
    except IntegrityError as error:
        db.session.expire_all()
        raise InvitationRejected(
            "identite_en_cours",
            "Cette acceptation est déjà en cours de traitement.",
            status=409,
        ) from error
    logger.info("invitation_chauffeur_acceptee public_id=%s", row.public_id)
    driver_id = int(created.id) if created is not None else None
    if driver_id is None:
        existing = Driver.query.filter_by(user_id=int(user.id)).one_or_none()
        if existing is not None:
            driver_id = int(existing.id)
    if driver_id is not None:
        try:
            from routes.auth import _save_active_context

            _save_active_context(str(user.public_id), f"driver:{driver_id}")
        except Exception:
            logger.warning(
                "contexte_chauffeur_non_persiste public_id=%s driver_id=%s",
                user.public_id,
                driver_id,
            )
    return {
        "status": "accepted",
        "already_accepted": False,
        "notification": None,
        "activation": _activation_state(),
        "company_name": _company_name(row),
        "portal_open": _driver_portal_open(user, int(row.company_id)),
        "active_context_id": f"driver:{driver_id}" if driver_id is not None else None,
    }


def _guard(company: Company, actor: User) -> None:
    if int(company.user_id) != int(actor.id):
        raise InvitationRejected(
            "acteur_non_autorise",
            "Seul le responsable de l'entreprise peut inviter un chauffeur.",
            status=403,
        )
    if not bool(company.is_approved):
        raise InvitationRejected(
            "entreprise_non_approuvee",
            "Une entreprise non approuvée ne peut pas inviter de chauffeur.",
            status=403,
        )


def _owned(company: Company, public_id: str, *, lock: bool) -> DriverInvitation:
    query = DriverInvitation.query.filter_by(
        company_id=int(company.id),
        public_id=str(public_id),
    )
    if lock:
        query = query.with_for_update()
    row = query.one_or_none()
    if row is None:
        raise InvitationRejected(
            "invitation_introuvable",
            "Invitation introuvable.",
            status=404,
        )
    return row


def _by_token(raw_token: str, *, lock: bool) -> DriverInvitation | None:
    secret = str(raw_token or "").strip()
    if not secret or len(secret) > 200:
        return None
    query = DriverInvitation.query.filter_by(token_hash=_hash_token(secret))
    if lock:
        query = query.with_for_update()
    return query.one_or_none()


def _attach(
    row: DriverInvitation, account: User | None, payload: dict[str, Any]
) -> tuple[User, Driver | None]:
    if account is None:
        account = _create_client(row, str(payload.get("password") or ""))
        driver = _create_company_driver(account, int(row.company_id))
        return account, driver
    driver = Driver.query.filter_by(user_id=int(account.id)).one_or_none()
    if driver is None:
        return account, _create_company_driver(account, int(row.company_id))
    return account, None


def _create_client(row: DriverInvitation, password: str) -> User:
    user = User()
    user.username = _username()
    user.first_name = row.first_name
    user.last_name = row.last_name
    user.email = row.email
    user.role = UserRole.CLIENT
    user.public_id = str(uuid.uuid4())
    user.account_status = "active"
    user.force_password_change = False
    _validate_chosen_password(password)
    # Mot de passe déjà passé par _validate_chosen_password (semgrep Django FP).
    # nosemgrep: python.django.security.audit.unvalidated-password.unvalidated-password
    user.set_password(password)
    db.session.add(user)
    db.session.flush()
    return user


def _create_company_driver(user: User, company_id: int) -> Driver:
    """Rattache un chauffeur actif à une seule entreprise.

    Aucune règle métier n'exige de le laisser inactif. La disponibilité
    dispatch reste fermée : elle commence quand le chauffeur prend son service.
    """
    driver = Driver()
    driver.user_id = int(user.id)
    driver.company_id = int(company_id)
    driver.is_active = True
    driver.is_available = False
    driver.driver_type = DriverType.REGULAR
    db.session.add(driver)
    db.session.flush()
    return driver


def _ensure_driver_grant(user: User, company_id: int) -> None:
    """Habilitation chauffeur limitée à l'entreprise du lien."""
    if not enforcement_enabled():
        raise InvitationRejected(
            "acces_non_ouvert",
            "L'habilitation chauffeur n'est pas ouverte dans ce processus.",
            status=409,
        )
    existing = PortalGrant.query.filter_by(
        user_id=int(user.id),
        portal=PortalKind.DRIVER.value,
        company_id=int(company_id),
    ).one_or_none()
    if existing is not None and existing.status == PortalGrantStatus.REVOKED.value:
        raise InvitationRejected(
            "habilitation_revoquee",
            "Une habilitation chauffeur révoquée ne peut pas être recréée.",
            status=409,
        )
    if existing is None:
        existing = PortalGrant(
            user_id=int(user.id),
            portal=PortalKind.DRIVER.value,
            company_id=int(company_id),
            status=PortalGrantStatus.ACTIVE.value,
        )
        db.session.add(existing)
    else:
        existing.status = PortalGrantStatus.ACTIVE.value
        existing.revoked_at = None
    db.session.flush()


def _remove_created_driver(row: DriverInvitation) -> bool:
    if row.created_driver_id is None:
        return False
    _revoke_created_grant(row)
    driver = db.session.get(Driver, int(row.created_driver_id))
    if driver is None:
        return False
    if int(driver.company_id) != int(row.company_id):
        return False
    if row.accepted_user_id is not None and int(driver.user_id) != int(
        row.accepted_user_id
    ):
        return False
    db.session.delete(driver)
    return True


def _revoke_created_grant(row: DriverInvitation) -> None:
    if row.accepted_user_id is None:
        return
    grant = PortalGrant.query.filter_by(
        user_id=int(row.accepted_user_id),
        portal=PortalKind.DRIVER.value,
        company_id=int(row.company_id),
    ).one_or_none()
    if grant is None or grant.status == PortalGrantStatus.REVOKED.value:
        return
    grant.status = PortalGrantStatus.REVOKED.value
    grant.revoked_at = datetime.now(UTC)


def _require_same_account(account: User, authenticated_user_id: int | None) -> None:
    if authenticated_user_id is None:
        raise InvitationRejected(
            "authentification_requise",
            "Connectez-vous avec le compte destinataire. "
            "L'adresse e-mail seule ne rattache pas le compte.",
            status=401,
        )
    if int(authenticated_user_id) != int(account.id):
        raise InvitationRejected(
            "destinataire_different",
            "Le compte connecté n'est pas le destinataire de cette invitation.",
            status=403,
        )


def _refuse_other_company(account: User, company_id: int) -> None:
    driver = Driver.query.filter_by(user_id=int(account.id)).one_or_none()
    if driver is not None and int(driver.company_id) != int(company_id):
        raise InvitationRejected(
            "deja_rattache",
            "Ce compte est déjà rattaché à une autre entreprise.",
            status=409,
        )


def _require_new_password(row: DriverInvitation, payload: dict[str, Any]) -> None:
    if int(row.attempt_count) >= MAX_ATTEMPTS:
        raise InvitationRejected(
            "tentatives_limitees",
            "Trop de tentatives. Demandez un nouvel envoi.",
            status=429,
        )
    password = str(payload.get("password") or "")
    try:
        _validate_chosen_password(password)
    except ValueError as error:
        row.attempt_count = int(row.attempt_count) + 1
        _persist()
        if int(row.attempt_count) >= MAX_ATTEMPTS:
            raise InvitationRejected(
                "tentatives_limitees",
                "Trop de tentatives. Demandez un nouvel envoi.",
                status=429,
            ) from error
        raise InvitationRejected(
            "mot_de_passe_refuse",
            str(error) or "Le mot de passe ne respecte pas les exigences.",
        ) from error


def _validate_chosen_password(password: str) -> None:
    try:
        PasswordPolicyService.validate_password(
            password, user_id=None, check_history=False
        )
    except PasswordPolicyError as error:
        raise ValueError(str(error)) from error


def _account_for(email: str) -> User | None:
    return User.query.filter(func.lower(User.email) == email).one_or_none()


def _mark_expired(row: DriverInvitation) -> bool:
    if row.status != "pending":
        return False
    expires_at = row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at > datetime.now(UTC):
        return False
    row.status = "expired"
    return True


def _company_item(row: DriverInvitation) -> dict[str, Any]:
    return {
        "public_id": row.public_id,
        "email": row.email,
        "first_name": row.first_name,
        "last_name": row.last_name,
        "status": row.status,
        "status_label": _STATUS_LABELS.get(row.status, row.status),
        "expires_at": _iso(row.expires_at),
        "created_at": _iso(row.created_at),
        "accepted_at": _iso(row.accepted_at),
        "resend_count": int(row.resend_count),
    }


def _public_view(row: DriverInvitation) -> dict[str, Any]:
    account = _account_for(row.email)
    return {
        "status": row.status,
        "status_label": _STATUS_LABELS.get(row.status, row.status),
        "company_name": _company_name(row),
        "first_name": row.first_name,
        "email_hint": _mask_email(row.email),
        "identity": "account_present" if account is not None else "new",
        "activation": _activation_state(),
        "expires_at": _iso(row.expires_at),
    }


def _activation_state() -> dict[str, bool]:
    return {"enabled": driver_invitation_activation_enabled()}


def _company_name(row: DriverInvitation) -> str:
    company = row.company or db.session.get(Company, int(row.company_id))
    if company is None:
        return "Entreprise"
    return str(company.name)


def _driver_portal_open(user: User, company_id: int) -> bool:
    return portal_company_allowed(user, PortalKind.DRIVER.value, int(company_id))


def _notify(row: DriverInvitation, raw_token: str, template: str) -> dict[str, Any]:
    body = (
        f"{_company_name(row)} vous invite à rejoindre son équipe. "
        "Le lien est personnel et ne concerne que cette entreprise."
    )
    if current_app.config.get("TESTING"):
        current_app.config.setdefault("_DRIVER_INVITE_LINKS", []).append(
            {"email": row.email, "token": raw_token, "template": template}
        )
        current_app.config.setdefault("_DRIVER_INVITE_NOTICES", []).append(
            {"email": row.email, "template": template, "body": body}
        )
        return {"state": "queued", "delivered": False, "template": template}
    return _hand_off(row.email, template, body, raw_token, _company_name(row))


def _hand_off(
    email: str, template: str, body: str, raw_token: str, company_name: str
) -> dict[str, Any]:
    if current_app.config.get("MAIL_SUPPRESS_SEND", True):
        return {"state": "queued", "delivered": False, "template": template}
    domain = email.rsplit("@", 1)[-1].lower()
    if domain not in CONTROLLED_DOMAINS:
        return {"state": "withheld", "delivered": False, "template": template}
    sender = (
        current_app.config.get("MAIL_DEFAULT_SENDER")
        or current_app.config.get("MAIL_USERNAME")
        or "noreply@lirie.ch"
    )
    link = f"{_frontend_base()}/invitation/chauffeur?token={raw_token}"
    message = Message(
        subject=f"Invitation chauffeur — {company_name}",
        sender=sender,
        recipients=[email],
        body=(
            f"{body}\n\n{link}\n\n"
            "Ce lien est à usage unique. "
            "Aucun mot de passe ne vous est transmis : vous choisissez le vôtre. "
            "L'activation du portail chauffeur reste fermée tant que LIRIE "
            "ne l'a pas autorisée."
        ),
    )
    try:
        mail.send(message)
    except Exception:
        logger.warning("invitation_chauffeur_envoi_impossible")
        return {"state": "failed", "delivered": False, "template": template}
    return {"state": "handed_to_mailer", "delivered": False, "template": template}


def _frontend_base() -> str:
    configured = str(current_app.config.get("FRONTEND_URL") or "").rstrip("/")
    if configured:
        return configured
    return "http://localhost:3000"


def _persist() -> None:
    db.session.flush()
    if not current_app.config.get("TESTING"):
        db.session.commit()


def _hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode("utf-8")).hexdigest()


def _new_secret() -> str:
    return secrets.token_urlsafe(32)


def _reject_forbidden(payload: dict[str, Any], forbidden: frozenset[str]) -> None:
    if not isinstance(payload, dict):
        raise InvitationRejected("payload_invalide", "La demande est incomplète.")
    if forbidden.intersection(payload):
        raise InvitationRejected(
            "droit_refuse",
            "Cette demande ne peut pas attribuer de droit professionnel.",
        )


def _email(value: Any) -> str:
    email = str(value or "").strip().lower()
    if not _is_valid_email(email):
        raise InvitationRejected(
            "email_invalide",
            "Indiquez une adresse e-mail valable.",
        )
    return email


def _name(value: Any, label: str) -> str:
    name = str(value or "").strip()
    if not name:
        raise InvitationRejected("champ_manquant", f"Indiquez {label}.")
    if len(name) > 80:
        raise InvitationRejected("champ_trop_long", f"Le champ {label} est trop long.")
    return name


def _username() -> str:
    for _ in range(5):
        candidate = f"chauffeur_{uuid.uuid4().hex[:12]}"
        if User.query.filter_by(username=candidate).one_or_none() is None:
            return candidate
    return f"chauffeur_{uuid.uuid4().hex}"


def _mask_email(email: str) -> str:
    local, _, domain = email.partition("@")
    if not local or not domain:
        return "***"
    return f"{local[0]}***@{domain}"


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()
