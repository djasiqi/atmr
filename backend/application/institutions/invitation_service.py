"""Service d'invitation et gestion des identités institutionnelles.

Gère:
- Génération de tokens d'invitation (secrets.token_urlsafe + sha256)
- Envoi d'email via Brevo provider existant
- Rattachement utilisateurs existants + notification d'accès
- Création Mode B (identifiant composé + mot de passe temporaire)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from flask import current_app, render_template

from ext import db
from models import Institution, User, UserRole
from services.email.brevo_provider import BrevoEmailProvider

logger = logging.getLogger(__name__)

INVITE_TOKEN_BYTES = 32
INVITE_EXPIRY_HOURS = 48
TEMP_PASSWORD_EXPIRY_DAYS = 14

ROLE_LABELS = {
    "institution_admin": "Administrateur",
    "institution_requester": "Demandeur",
    "institution_reader": "Lecteur",
    "institution_billing": "Facturation",
    "institution_curator": "Curateur",
    "institution_reception": "Réception",
}

InvitePath = Literal[
    "conflict_same_institution",
    "conflict_other_institution",
    "existing_user",
    "new_user",
    "new_username",
    "username_reserved",
    "validation_error",
]

EmailType = Literal["invitation", "access_notification"] | None


@dataclass
class InviteResult:
    """Résultat d'envoi d'email."""

    success: bool
    token: str | None = None
    error: str | None = None


@dataclass
class InviteAttachResult:
    """Résultat de invite_or_attach_institution_user."""

    path: InvitePath
    http_status: int
    message: str | None = None
    error: str | None = None
    user: User | None = None
    raw_token: str | None = None
    email_result: InviteResult | None = None
    email_type: EmailType = None
    temporary_credentials: dict[str, str] | None = None
    credentials_shown_once: bool = False
    creation_mode: str = "email"
    audit_details: dict[str, Any] = field(default_factory=dict)


_EMAIL_ERROR_KEYWORDS: dict[str, str] = {
    "timeout": "Délai d'attente dépassé (timeout)",
    "timed out": "Délai d'attente dépassé (timeout)",
    "connection refused": "Impossible de contacter le service d'envoi",
    "connect": "Impossible de contacter le service d'envoi",
    "401": "Erreur d'authentification du service email",
    "403": "Erreur d'authentification du service email",
    "unauthorized": "Erreur d'authentification du service email",
    "forbidden": "Erreur d'authentification du service email",
    "smtp": "Erreur du serveur d'envoi",
    "relay": "Erreur du serveur d'envoi",
}


def _sanitize_email_error(raw_error: str | None) -> str:
    if not raw_error:
        return "Erreur inconnue lors de l'envoi"
    lower = raw_error.lower()
    for keyword, safe_msg in _EMAIL_ERROR_KEYWORDS.items():
        if keyword in lower:
            return safe_msg
    if "rate" in lower and "limit" in lower:
        return "Limite d'envoi atteinte, réessayez plus tard"
    if "invalid" in lower and ("email" in lower or "address" in lower):
        return "Adresse email invalide ou rejetée"
    return "Erreur lors de l'envoi de l'email"


def _normalize_job_title(value: str | None) -> str | None:
    """Normalise une fonction/metier : trim + collapse des espaces internes.

    Evite que 'Infirmier   diplome' et '  Infirmier diplome ' deviennent
    des valeurs distinctes. Retourne None si vide.
    """
    if not value:
        return None
    collapsed = re.sub(r"\s+", " ", str(value)).strip()
    return collapsed or None


def generate_invite_token() -> tuple[str, str]:
    raw_token = secrets.token_urlsafe(INVITE_TOKEN_BYTES)
    token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    return raw_token, token_hash


def hash_token(raw_token: str) -> str:
    return hashlib.sha256(raw_token.encode()).hexdigest()


def get_invite_expiry() -> datetime:
    return datetime.now(UTC) + timedelta(hours=INVITE_EXPIRY_HOURS)


def _frontend_url() -> str:
    return os.getenv("FRONTEND_URL", "http://localhost:3000").rstrip("/")


def build_invite_url(raw_token: str) -> str:
    return f"{_frontend_url()}/invite/{raw_token}"


def build_login_url() -> str:
    return f"{_frontend_url()}/login"


def get_role_label(role: str) -> str:
    return ROLE_LABELS.get(role, role or "Inconnu")


def _send_html_email(
    *,
    to_email: str,
    to_name: str,
    subject: str,
    template_name: str,
    template_context: dict[str, Any],
    log_prefix: str,
    raw_token: str | None = None,
) -> InviteResult:
    try:
        with current_app.app_context():
            html_content = render_template(template_name, **template_context)

        provider = BrevoEmailProvider()
        from_email = os.getenv("INVITE_FROM_EMAIL", "noreply@lirie.ch")
        from_name = os.getenv("INVITE_FROM_NAME", "Lirie - Portail Institution")

        result = provider.send_invoice_email(
            from_email=from_email,
            from_name=from_name,
            to_email=to_email,
            to_name=to_name,
            subject=subject,
            html_content=html_content,
        )

        if result.success:
            logger.info("%s Email envoyé à %s", log_prefix, to_email)
            return InviteResult(success=True, token=raw_token)

        logger.error("%s Échec envoi email à %s: %s", log_prefix, to_email, result.error)
        return InviteResult(success=False, error=_sanitize_email_error(result.error))
    except Exception as e:
        logger.exception("%s Erreur envoi email à %s: %s", log_prefix, to_email, e)
        return InviteResult(success=False, error=_sanitize_email_error(str(e)))


def send_invitation_email(
    to_email: str,
    first_name: str | None,
    institution_name: str,
    inviter_name: str,
    role: str,
    raw_token: str,
) -> InviteResult:
    invite_url = build_invite_url(raw_token)
    return _send_html_email(
        to_email=to_email,
        to_name=first_name or to_email.split("@")[0],
        subject=f"Invitation - {institution_name}",
        template_name="emails/invitation_email.html",
        template_context={
            "institution_name": institution_name,
            "first_name": first_name,
            "inviter_name": inviter_name,
            "role_label": get_role_label(role),
            "invite_url": invite_url,
            "current_year": datetime.now(UTC).year,
        },
        log_prefix="[Invitation]",
        raw_token=raw_token,
    )


def send_institution_access_email(
    to_email: str,
    first_name: str | None,
    institution_name: str,
    inviter_name: str,
    role: str,
) -> InviteResult:
    return _send_html_email(
        to_email=to_email,
        to_name=first_name or to_email.split("@")[0],
        subject=f"Accès institution - {institution_name}",
        template_name="emails/institution_access_email.html",
        template_context={
            "institution_name": institution_name,
            "first_name": first_name,
            "inviter_name": inviter_name,
            "role_label": get_role_label(role),
            "login_url": build_login_url(),
            "current_year": datetime.now(UTC).year,
        },
        log_prefix="[Institution Access]",
    )


def _preserve_or_set_institution_role(existing_user: User) -> None:
    if existing_user.role in (None, UserRole.INSTITUTION):
        existing_user.role = UserRole.INSTITUTION


def _generate_strong_password(*, length: int = 16) -> str:
    import string

    upper = secrets.choice(string.ascii_uppercase)
    lower = secrets.choice(string.ascii_lowercase)
    digit = secrets.choice(string.digits)
    special = secrets.choice("!@#$%^&*()-_=+")
    remaining = [
        secrets.choice(string.ascii_letters + string.digits + "!@#$%^&*()-_=+")
        for _ in range(max(length - 4, 8))
    ]
    chars = [upper, lower, digit, special, *remaining]
    secrets.SystemRandom().shuffle(chars)
    return "".join(chars)


def invite_or_attach_institution_user(
    *,
    institution: Institution,
    admin_user: User,
    email: str | None = None,
    role_value: str,
    first_name: str | None = None,
    last_name: str | None = None,
    creation_mode: str = "email",
    local_username: str | None = None,
    job_title: str | None = None,
) -> InviteAttachResult:
    """Unifie invitation email, notification d'accès et création Mode B."""
    institution_name = str(institution.name or "Institution")
    admin_name = str(admin_user.full_name or admin_user.email or "L'administrateur")
    job_title = _normalize_job_title(job_title)

    logger.info(
        "[Institution Invite] email=%s path=pending mode=%s username=%s",
        email or "",
        creation_mode,
        local_username or "",
    )

    if creation_mode == "username":
        from application.users.normalization import normalize_contact_email

        contact_email = normalize_contact_email(email)
        return _create_username_mode_user(
            institution=institution,
            admin_user=admin_user,
            role_value=role_value,
            first_name=first_name,
            last_name=last_name,
            local_username=str(local_username or "").strip().lower(),
            admin_name=admin_name,
            institution_name=institution_name,
            job_title=job_title,
            contact_email=contact_email,
        )

    if not email:
        return InviteAttachResult(
            path="validation_error",
            http_status=400,
            error="L'email est requis",
        )

    email = email.strip().lower()
    from sqlalchemy import func

    existing_user = User.query.filter(func.lower(User.email) == email).first()

    if existing_user:
        if existing_user.institution_id == institution.id:
            logger.info(
                "[Institution Invite] email=%s path=conflict_same_institution mode=%s",
                email,
                creation_mode,
            )
            error_msg = (
                "Cet utilisateur est désactivé. Utilisez 'Renvoyer invitation' pour le réactiver."
                if existing_user.account_status == "disabled"
                else "Cet utilisateur fait déjà partie de l'institution"
            )
            return InviteAttachResult(
                path="conflict_same_institution",
                http_status=409,
                error=error_msg,
            )

        if existing_user.institution_id is not None:
            logger.info(
                "[Institution Invite] email=%s path=conflict_other_institution mode=%s",
                email,
                creation_mode,
            )
            return InviteAttachResult(
                path="conflict_other_institution",
                http_status=409,
                error="Cet utilisateur appartient déjà à une autre institution",
            )

        logger.info(
            "[Institution Invite] email=%s path=existing_user mode=%s",
            email,
            creation_mode,
        )

        existing_user.institution_id = institution.id
        existing_user.institution_role = role_value
        _preserve_or_set_institution_role(existing_user)
        existing_user.account_status = "active"
        if hasattr(existing_user, "authentication_method"):
            existing_user.authentication_method = "email"
        if first_name:
            existing_user.first_name = first_name
        if last_name:
            existing_user.last_name = last_name
        if job_title is not None:
            existing_user.job_title = job_title

        db.session.commit()

        access_result = dispatch_institution_email(
            email_type="access_notification",
            to_email=email,
            first_name=str(first_name) if first_name else None,
            institution_name=institution_name,
            inviter_name=admin_name,
            role=role_value,
            user_id=existing_user.id,
            path="existing_user",
        )

        if not access_result.success:
            logger.warning(
                "[Institution] Utilisateur ajouté mais notification non envoyée: %s",
                access_result.error,
            )

        message = (
            "Notification d'accès envoyée par email"
            if access_result.success
            else "Utilisateur ajouté mais la notification n'a pas pu être envoyée"
        )

        return InviteAttachResult(
            path="existing_user",
            http_status=200 if access_result.success else 207,
            message=message,
            user=existing_user,
            email_result=access_result,
            email_type="access_notification",
            creation_mode="email",
            audit_details={
                "target_user_id": existing_user.id,
                "target_email": email,
                "institution_role": role_value,
                "method": "existing_user",
                "email_sent": access_result.success,
            },
        )

    logger.info(
        "[Institution Invite] email=%s path=new_user mode=%s",
        email,
        creation_mode,
    )

    raw_token, token_hash = generate_invite_token()
    placeholder_password = secrets.token_urlsafe(32)

    new_user = User()
    new_user.public_id = str(uuid.uuid4())
    new_user.username = email
    new_user.email = email
    new_user.first_name = first_name or None
    new_user.last_name = last_name or None
    new_user.job_title = job_title
    new_user.role = UserRole.INSTITUTION
    new_user.institution_id = institution.id
    new_user.institution_role = role_value
    new_user.account_status = "invited"
    new_user.invite_token_hash = token_hash
    new_user.invite_expires_at = get_invite_expiry()
    new_user.invite_sent_at = datetime.now(UTC)
    new_user.force_password_change = True
    if hasattr(new_user, "authentication_method"):
        new_user.authentication_method = "email"
    new_user.set_password(placeholder_password, force_change=True)

    db.session.add(new_user)
    db.session.commit()

    invite_result = dispatch_institution_email(
        email_type="invitation",
        to_email=email,
        first_name=str(first_name) if first_name else None,
        institution_name=institution_name,
        inviter_name=admin_name,
        role=role_value,
        raw_token=raw_token,
        user_id=new_user.id,
        path="new_user",
    )

    if not invite_result.success:
        logger.warning(
            "[Institution] Invitation créée mais email non envoyé: %s",
            invite_result.error,
        )

    message = (
        "Invitation envoyée par email"
        if invite_result.success
        else "Utilisateur créé mais l'email n'a pas pu être envoyé"
    )

    return InviteAttachResult(
        path="new_user",
        http_status=201,
        message=message,
        user=new_user,
        raw_token=raw_token,
        email_result=invite_result,
        email_type="invitation",
        creation_mode="email",
        audit_details={
            "target_user_id": new_user.id,
            "target_email": email,
            "institution_role": role_value,
            "method": "email_invitation",
            "email_sent": invite_result.success,
        },
    )


def _create_username_mode_user(
    *,
    institution: Institution,
    admin_user: User,
    role_value: str,
    first_name: str | None,
    last_name: str | None,
    local_username: str,
    admin_name: str,
    institution_name: str,
    job_title: str | None = None,
    contact_email: str | None = None,
) -> InviteAttachResult:
    local_username = local_username.strip().lower()
    job_title = _normalize_job_title(job_title)

    if not local_username or len(local_username) < 3:
        return InviteAttachResult(
            path="validation_error",
            http_status=400,
            error="L'identifiant doit contenir au moins 3 caractères",
        )

    from sqlalchemy import func

    conflict = User.query.filter(func.lower(User.username) == local_username).first()
    if conflict:
        return InviteAttachResult(
            path="conflict_username",
            http_status=409,
            error="Cet identifiant est déjà utilisé sur la plateforme",
        )

    if contact_email:
        from application.users.normalization import find_user_by_normalized_email

        email_conflict = find_user_by_normalized_email(contact_email)
        if email_conflict:
            return InviteAttachResult(
                path="conflict_email",
                http_status=409,
                error="Cet email est déjà utilisé sur la plateforme",
            )

    temp_password = _generate_strong_password()
    now = datetime.now(UTC)

    new_user = User()
    new_user.public_id = str(uuid.uuid4())
    new_user.username = local_username
    new_user.email = contact_email
    new_user.first_name = first_name or None
    new_user.last_name = last_name or None
    new_user.job_title = job_title
    new_user.role = UserRole.INSTITUTION
    new_user.institution_id = institution.id
    new_user.institution_role = role_value
    new_user.account_status = "active"
    new_user.force_password_change = True
    if hasattr(new_user, "authentication_method"):
        new_user.authentication_method = "username"
    if hasattr(new_user, "password_expires_at"):
        new_user.password_expires_at = now + timedelta(days=TEMP_PASSWORD_EXPIRY_DAYS)
    if hasattr(new_user, "temporary_password_created_at"):
        new_user.temporary_password_created_at = now
    if hasattr(new_user, "last_password_reset_at"):
        new_user.last_password_reset_at = now
    if hasattr(new_user, "temp_password_generation_count"):
        new_user.temp_password_generation_count = 1
    if hasattr(new_user, "first_login_completed_at"):
        new_user.first_login_completed_at = None
    new_user.set_password(temp_password, force_change=True)

    db.session.add(new_user)
    db.session.commit()

    logger.info(
        "[Institution Invite] path=new_username username=%s institution_id=%s",
        local_username,
        institution.id,
    )

    return InviteAttachResult(
        path="new_username",
        http_status=201,
        message="Utilisateur créé avec identifiant institutionnel",
        user=new_user,
        email_type=None,
        creation_mode="username",
        temporary_credentials={
            "username": local_username,
            "temporary_password": temp_password,
        },
        credentials_shown_once=True,
        audit_details={
            "target_user_id": new_user.id,
            "institution_role": role_value,
            "method": "username_creation",
            "creation_mode": "username",
        },
    )


def _inc_institution_metric(*, path: str, email_type: str, result: str) -> None:
    try:
        from security.institution_metrics import institution_invitations_total

        if institution_invitations_total is not None:
            institution_invitations_total.labels(
                path=path, email_type=email_type or "none", result=result
            ).inc()
    except Exception:
        pass


def dispatch_institution_email(
    *,
    email_type: str,
    to_email: str,
    first_name: str | None,
    institution_name: str,
    inviter_name: str,
    role: str,
    raw_token: str | None = None,
    user_id: int | None = None,
    path: str = "unknown",
) -> InviteResult:
    """Enqueue Celery ou fallback synchrone."""
    try:
        from tasks.institution_invitation_tasks import send_institution_email_task

        send_institution_email_task.delay(
            email_type=email_type,
            to_email=to_email,
            first_name=first_name,
            institution_name=institution_name,
            inviter_name=inviter_name,
            role=role,
            raw_token=raw_token,
            user_id=user_id,
        )
        _inc_institution_metric(path=path, email_type=email_type, result="enqueued")
        return InviteResult(success=True, token=raw_token)
    except Exception as enqueue_err:
        logger.warning(
            "[Institution Invite] Celery indisponible, envoi synchrone: %s",
            enqueue_err,
        )

    if email_type == "access_notification":
        result = send_institution_access_email(
            to_email=to_email,
            first_name=first_name,
            institution_name=institution_name,
            inviter_name=inviter_name,
            role=role,
        )
    else:
        if not raw_token:
            return InviteResult(success=False, error="Token d'invitation manquant")
        result = send_invitation_email(
            to_email=to_email,
            first_name=first_name,
            institution_name=institution_name,
            inviter_name=inviter_name,
            role=role,
            raw_token=raw_token,
        )

    _inc_institution_metric(
        path=path,
        email_type=email_type,
        result="sent" if result.success else "failed",
    )
    return result
