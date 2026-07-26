"""Livraison email d'activation Lot 1 : HMAC, historique, finalisation idempotente."""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import text

from ext import db
from models.activation_email_delivery import (
    ALLOWED_TRANSITIONS,
    EMAIL_DELIVERY_FAILED,
    EMAIL_DELIVERY_KIND_INITIAL,
    EMAIL_DELIVERY_KIND_RESEND,
    EMAIL_DELIVERY_QUEUED,
    EMAIL_DELIVERY_SENDING,
    EMAIL_DELIVERY_SENT,
    SENDING_LEASE_MINUTES,
    WEBHOOK_ADVANCED_STATUSES,
    ActivationEmailDelivery,
)
from models.activation_session import ActivationSession
from services.notifications.activation_token import (
    ActivationTokenKeyError,
    derive_activation_token,
    hash_activation_token,
    require_activation_token_key_in_production,
)
from services.notifications.email import is_email_provider_configured

logger = logging.getLogger(__name__)

ACTIVATION_EMAIL_TTL_MINUTES = int(os.getenv("ACTIVATION_EMAIL_TTL_MINUTES", "30"))
CURRENT_TOKEN_KEY_VERSION = 1


def sanitize_email_error(message: str | None) -> str:
    """Supprime jetons, URLs et emails avant stockage interne."""
    if not message:
        return "email_send_failed"
    text_val = str(message)
    text_val = re.sub(r"https?://\S+", "[url]", text_val)
    text_val = re.sub(r"[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-.]+", "[token]", text_val)
    text_val = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "[email]",
        text_val,
    )
    return text_val[:500]


def _sync_session_from_delivery(
    session: ActivationSession, delivery: ActivationEmailDelivery
) -> None:
    """Maintient les colonnes miroir sur ActivationSession (courant)."""
    session.email_delivery_id = delivery.email_delivery_id
    session.email_delivery_status = delivery.status
    session.email_delivery_kind = delivery.kind
    session.email_token_hash = delivery.email_token_hash
    session.email_token_expires_at = delivery.token_expires_at
    session.email_last_error = delivery.last_error
    session.email_provider_message_id = delivery.provider_message_id


def get_delivery_by_id(email_delivery_id: str) -> ActivationEmailDelivery | None:
    return ActivationEmailDelivery.query.filter_by(
        email_delivery_id=email_delivery_id
    ).first()


def is_sending_lease_expired(delivery: ActivationEmailDelivery) -> bool:
    if delivery.status != EMAIL_DELIVERY_SENDING:
        return False
    started = delivery.sending_started_at
    if started is None:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=UTC)
    return datetime.now(UTC) - started > timedelta(minutes=SENDING_LEASE_MINUTES)


def can_start_new_delivery(session: ActivationSession) -> tuple[bool, str | None]:
    """False si livraison courante encore queued/sending (lease non expirée)."""
    if not session.email_delivery_id:
        return True, None
    delivery = get_delivery_by_id(session.email_delivery_id)
    if delivery is None:
        return True, None
    if delivery.status == EMAIL_DELIVERY_QUEUED:
        return False, "email_delivery_in_progress"
    if delivery.status == EMAIL_DELIVERY_SENDING and not is_sending_lease_expired(
        delivery
    ):
        return False, "email_delivery_in_progress"
    if delivery.status == EMAIL_DELIVERY_SENDING and is_sending_lease_expired(delivery):
        delivery.status = EMAIL_DELIVERY_FAILED
        delivery.last_error = sanitize_email_error("sending_lease_expired")
        _sync_session_from_delivery(session, delivery)
        return True, None
    return True, None


def mark_delivery_failed(
    session: ActivationSession,
    error: str,
    *,
    email_delivery_id: str | None = None,
) -> None:
    """Marque l'envoi courant (et la ligne livraison) en failed."""
    delivery_id = email_delivery_id or session.email_delivery_id
    if (
        email_delivery_id
        and session.email_delivery_id
        and session.email_delivery_id != email_delivery_id
    ):
        return
    session.email_delivery_status = EMAIL_DELIVERY_FAILED
    session.email_last_error = sanitize_email_error(error)
    if delivery_id:
        delivery = get_delivery_by_id(delivery_id)
        if delivery:
            delivery.status = EMAIL_DELIVERY_FAILED
            delivery.last_error = session.email_last_error


def prepare_activation_email_delivery(
    session: ActivationSession,
    *,
    kind: str,
) -> tuple[str, str]:
    """Crée une livraison + jeton HMAC. Retourne (delivery_id, token)."""
    if kind not in {EMAIL_DELIVERY_KIND_INITIAL, EMAIL_DELIVERY_KIND_RESEND}:
        raise ValueError(f"kind invalide: {kind}")
    require_activation_token_key_in_production()

    delivery_id = str(uuid.uuid4())
    key_version = CURRENT_TOKEN_KEY_VERSION
    token = derive_activation_token(delivery_id, key_version=key_version)
    token_hash = hash_activation_token(token)
    now = datetime.now(UTC)
    expires = now + timedelta(minutes=ACTIVATION_EMAIL_TTL_MINUTES)

    delivery = ActivationEmailDelivery(
        activation_session_pk=session.id,
        email_delivery_id=delivery_id,
        kind=kind,
        status=EMAIL_DELIVERY_QUEUED,
        token_key_version=key_version,
        email_token_hash=token_hash,
        token_expires_at=expires,
    )
    db.session.add(delivery)
    _sync_session_from_delivery(session, delivery)
    session.email_last_error = None
    return delivery_id, token


def enqueue_activation_email(
    *,
    activation_session_id: str,
    email_delivery_id: str,
) -> None:
    from tasks.notification_tasks import send_activation_email_task

    send_activation_email_task.delay(
        activation_session_id=activation_session_id,
        email_delivery_id=email_delivery_id,
    )


def try_enqueue_activation_email(
    session: ActivationSession,
    *,
    kind: str,
    environment: str,
    is_testing: bool,
    email_token: str | None = None,
    email_token_hash: str | None = None,
) -> dict[str, Any]:
    """Prépare + enqueue. email_token/hash args ignorés (HMAC Lot 1) — compat appels.

    Returns:
        dict ok/queued/email_sent/debug_activation_link/error/require_502/email_token
    """
    del email_token, email_token_hash  # compat signature ancienne
    env = (environment or "").strip().lower()
    is_prod = env == "production"
    is_local_dev = env == "development" and not is_testing

    ok_new, block_reason = can_start_new_delivery(session)
    if not ok_new:
        return {
            "ok": False,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": None,
            "error": block_reason or "email_delivery_in_progress",
            "require_502": False,
            "email_token": None,
        }

    ready, config_error = is_email_provider_configured()
    if not ready and is_prod:
        mark_delivery_failed(session, config_error or "email_provider_not_configured")
        if not session.email_delivery_status:
            session.email_delivery_status = EMAIL_DELIVERY_FAILED
        db.session.commit()
        return {
            "ok": False,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": None,
            "error": config_error or "email_provider_not_configured",
            "require_502": True,
            "email_token": None,
        }

    delivery_id: str | None = None
    token: str | None = None
    try:
        if is_prod:
            require_activation_token_key_in_production()
        delivery_id, token = prepare_activation_email_delivery(session, kind=kind)
        db.session.commit()
    except ActivationTokenKeyError as e:
        db.session.rollback()
        return {
            "ok": False,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": None,
            "error": str(e),
            "require_502": True,
            "email_token": None,
        }
    except Exception as e:
        logger.exception("[activation_email] échec préparation: %s", e)
        try:
            mark_delivery_failed(session, str(e))
            db.session.commit()
        except Exception:
            db.session.rollback()
        debug_link = None
        if is_local_dev and token:
            frontend = (
                os.getenv("FRONTEND_URL")
                or os.getenv("PUBLIC_FRONTEND_URL")
                or "http://localhost:3000"
            ).rstrip("/")
            debug_link = f"{frontend}/activate-account?token={token}"
        return {
            "ok": is_local_dev,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": debug_link,
            "error": sanitize_email_error(str(e)),
            "require_502": not is_local_dev and not is_testing,
            "email_token": token if is_local_dev else None,
        }

    if not ready and is_local_dev and token:
        frontend = (
            os.getenv("FRONTEND_URL")
            or os.getenv("PUBLIC_FRONTEND_URL")
            or "http://localhost:3000"
        ).rstrip("/")
        return {
            "ok": True,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": f"{frontend}/activate-account?token={token}",
            "error": None,
            "require_502": False,
            "email_token": token,
        }

    try:
        enqueue_activation_email(
            activation_session_id=session.activation_session_id,
            email_delivery_id=delivery_id or "",
        )
    except Exception as e:
        logger.exception("[activation_email] échec Celery delay: %s", e)
        if delivery_id:
            purge_activation_email_token(delivery_id)
        mark_delivery_failed(session, str(e), email_delivery_id=delivery_id)
        db.session.commit()
        return {
            "ok": False,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": None,
            "error": sanitize_email_error(str(e)),
            "require_502": is_prod or (not is_local_dev and not is_testing),
            "email_token": None,
        }

    return {
        "ok": True,
        "queued": True,
        "email_sent": None,
        "debug_activation_link": None,
        "error": None,
        "require_502": False,
        "email_token": token,
    }


def cas_claim_sending(
    session: ActivationSession,
    email_delivery_id: str,
) -> str:
    """Claim atomique queued→sending sur la ligne livraison.

    Returns:
        'proceed' | 'ignore'
    """
    if session.email_verified_at is not None:
        return "ignore"
    delivery = get_delivery_by_id(email_delivery_id)
    if delivery is None:
        return "ignore"
    if session.email_delivery_id != email_delivery_id:
        return "ignore"
    if delivery.status == EMAIL_DELIVERY_SENT:
        return "ignore"
    if delivery.status in WEBHOOK_ADVANCED_STATUSES:
        return "ignore"
    if delivery.provider_accepted_at is not None:
        return "ignore"

    now = datetime.now(UTC)
    if delivery.status == EMAIL_DELIVERY_QUEUED:
        updated = ActivationEmailDelivery.query.filter_by(
            id=delivery.id,
            email_delivery_id=email_delivery_id,
            status=EMAIL_DELIVERY_QUEUED,
        ).update(
            {
                "status": EMAIL_DELIVERY_SENDING,
                "sending_started_at": now,
            },
            synchronize_session=False,
        )
        db.session.commit()
        if updated == 0:
            db.session.refresh(delivery)
            if delivery.status == EMAIL_DELIVERY_SENDING:
                _sync_session_from_delivery(session, delivery)
                db.session.commit()
                return "proceed"
            return "ignore"
        db.session.refresh(delivery)
        _sync_session_from_delivery(session, delivery)
        db.session.commit()
        return "proceed"

    if delivery.status == EMAIL_DELIVERY_SENDING:
        # Retry même delivery_id (acks_late) — même jeton HMAC
        return "proceed"

    return "ignore"


def resolve_activation_token_for_delivery(email_delivery_id: str) -> str | None:
    """Reconstruit le jeton HMAC pour retries Celery."""
    delivery = get_delivery_by_id(email_delivery_id)
    if delivery is None:
        return None
    try:
        return derive_activation_token(
            email_delivery_id, key_version=int(delivery.token_key_version or 1)
        )
    except ActivationTokenKeyError:
        logger.error(
            "[activation_email] clé manquante pour delivery_id=%s version=%s",
            email_delivery_id,
            delivery.token_key_version,
        )
        return None


# Compat tâches / anciens imports Redis
get_activation_email_token = resolve_activation_token_for_delivery


def purge_activation_email_token(email_delivery_id: str) -> None:
    """No-op Lot 1 (jeton HMAC dérivable, plus de Redis)."""
    del email_delivery_id


def finalize_after_provider_accepted(
    session: ActivationSession,
    *,
    email_delivery_id: str,
    message_id: str | None,
) -> bool:
    """Finalisation idempotente post-HTTP 201 (provider_accepted_at IS NULL).

    Returns:
        True si finalisation appliquée (première fois), False si déjà faite.
    """
    delivery = get_delivery_by_id(email_delivery_id)
    if delivery is None or session.email_delivery_id != email_delivery_id:
        return False

    now = datetime.now(UTC)
    # UPDATE conditionnel : ne touche compteurs qu'une fois
    result = db.session.execute(
        text(
            """
            UPDATE activation_email_deliveries
            SET
              status = CASE
                WHEN status = :sending THEN :sent
                ELSE status
              END,
              provider_message_id = COALESCE(provider_message_id, :mid),
              provider_accepted_at = :now,
              last_error = NULL,
              updated_at = :now
            WHERE email_delivery_id = :did
              AND provider_accepted_at IS NULL
            """
        ),
        {
            "sending": EMAIL_DELIVERY_SENDING,
            "sent": EMAIL_DELIVERY_SENT,
            "mid": message_id,
            "now": now,
            "did": email_delivery_id,
        },
    )
    if result.rowcount == 0:  # type: ignore[attr-defined]
        return False

    db.session.refresh(delivery)

    # Compteurs session — uniquement lors de la première finalisation
    previous_sent_at = session.last_email_sent_at
    if delivery.kind == EMAIL_DELIVERY_KIND_RESEND:

        def _same_day(a: datetime, b: datetime) -> bool:
            return a.astimezone(UTC).date() == b.astimezone(UTC).date()

        if previous_sent_at and not _same_day(previous_sent_at, now):
            session.resend_count_email = 1
        else:
            session.resend_count_email = int(session.resend_count_email or 0) + 1
    session.last_email_sent_at = now
    session.email_last_error = None
    _sync_session_from_delivery(session, delivery)
    return True


# Alias rétrocompat tests / tâches
def mark_delivery_sent(
    session: ActivationSession,
    *,
    email_delivery_id: str,
    message_id: str | None,
) -> None:
    finalize_after_provider_accepted(
        session,
        email_delivery_id=email_delivery_id,
        message_id=message_id,
    )


def apply_delivery_transition(
    delivery: ActivationEmailDelivery,
    new_status: str,
) -> bool:
    """Applique une transition selon ALLOWED_TRANSITIONS. True si appliquée."""
    current = delivery.status
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    if new_status not in allowed:
        return False
    delivery.status = new_status
    return True
