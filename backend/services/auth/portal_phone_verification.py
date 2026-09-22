"""Validation téléphone CLIENT/PORTAL — découplée de l'activation du compte.

Source de vérité : ``User.phone_verified_at``.
L'e-mail active le compte. Le SMS n'est exigé qu'avant le premier transport.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from models import ActivationSession, Client, User
from models.enums import ClientType, UserRole
from services.notifications.phone_e164 import normalize_e164_phone

logger = logging.getLogger(__name__)

PHONE_VERIFICATION_REQUIRED = "phone_verification_required"


class PortalPhoneVerificationRequired(Exception):
    """PORTAL : téléphone non vérifié, confirmation de transport interdite."""

    code = PHONE_VERIFICATION_REQUIRED

    def __init__(
        self,
        message: str = "Validez votre téléphone pour confirmer la demande de transport.",
    ) -> None:
        self.message = message
        super().__init__(message)


def assert_portal_can_confirm_transport(
    *,
    user_id: int,
    client: object,
    user_loader: Callable[[int], object | None] | None = None,
) -> None:
    """Autorité métier : PORTAL + ``phone_verified_at`` NULL ⇒ refus.

    Les flux Institution / TRANSPORT / création manuelle entreprise ne sont
    pas concernés (``is_portal_client`` est faux).
    """
    if not is_portal_client(client):
        return
    if user_loader is not None:
        user = user_loader(user_id)
    else:
        from ext import db

        user = db.session.get(User, user_id)
    if user_phone_is_verified(user):
        return
    raise PortalPhoneVerificationRequired()


def _enum_value(value: object) -> str:
    raw = value.value if hasattr(value, "value") else value
    return str(raw or "").strip().lower()


def is_portal_client_user(user: User | None) -> bool:
    """True uniquement pour un CLIENT rattaché à un profil PORTAL."""
    if user is None:
        return False
    if _enum_value(getattr(user, "role", None)) != _enum_value(UserRole.CLIENT):
        return False
    clients = list(getattr(user, "clients", None) or [])
    if not clients:
        clients = Client.query.filter_by(user_id=user.id).all()
    return any(is_portal_client(client) for client in clients)


def is_portal_client(client: object | None) -> bool:
    return _enum_value(getattr(client, "client_type", None)) == _enum_value(
        ClientType.PORTAL
    )


def latest_activation_session(user: User) -> ActivationSession | None:
    return (
        ActivationSession.query.filter_by(user_id=user.id)
        .order_by(ActivationSession.created_at.desc())
        .first()
    )


def portal_email_is_verified(user: User) -> bool:
    """E-mail confirmé via la session d'activation (pas via la présence d'un numéro)."""
    if not (getattr(user, "email", None) or "").strip():
        return False
    session = latest_activation_session(user)
    return bool(session and session.email_verified_at)


def user_phone_is_verified(user: User | None) -> bool:
    return bool(user and getattr(user, "phone_verified_at", None))


def promote_portal_account_after_email(
    user: User,
    session: ActivationSession | None = None,
) -> bool:
    """Passe pending_activation → active si l'e-mail est confirmé.

    Ne touche jamais ``phone_verified_at``. Idempotent.
    """
    if not is_portal_client_user(user):
        return False
    session = session or latest_activation_session(user)
    if not session or not session.email_verified_at:
        return False

    changed = False
    if getattr(user, "account_status", None) == "pending_activation":
        user.account_status = "active"
        changed = True
    for client in getattr(user, "clients", None) or []:
        if not getattr(client, "is_active", True):
            client.is_active = True
            changed = True
    if session.consumed_at is None:
        session.consumed_at = datetime.now(UTC)
        changed = True
    if changed:
        logger.info(
            "portal_account_promoted_after_email user_id=%s activation_session_id=%s",
            user.id,
            session.activation_session_id,
        )
    return changed


def maybe_promote_portal_account(user: User | None) -> bool:
    """Compatibilité comptes déjà email-vérifiés mais encore pending_activation."""
    if user is None or not is_portal_client_user(user):
        return False
    if getattr(user, "account_status", None) != "pending_activation":
        return False
    if not portal_email_is_verified(user):
        return False
    return promote_portal_account_after_email(user)


def apply_user_phone_change(user: User, new_phone: str | None) -> str | None:
    """Met à jour le numéro E.164 et révoque la vérif si le numéro change.

    Returns:
        Numéro normalisé, ou ``None`` si la valeur fournie est vide.
        Lève ``ValueError`` si le format est invalide.
    """
    raw = (new_phone or "").strip()
    if not raw:
        return None
    normalized = normalize_e164_phone(raw)
    if not normalized:
        raise ValueError("invalid_phone")
    previous = normalize_e164_phone(getattr(user, "phone", None))
    user.phone = normalized
    if previous != normalized:
        user.phone_verified_at = None
        logger.info(
            "phone_verification_revoked user_id=%s reason=phone_changed",
            user.id,
        )
    return normalized
