"""Livraison email d'activation : Redis jeton, enqueue Celery, compensations."""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from ext import db, redis_client
from models.activation_session import (
    EMAIL_DELIVERY_FAILED,
    EMAIL_DELIVERY_KIND_INITIAL,
    EMAIL_DELIVERY_KIND_RESEND,
    EMAIL_DELIVERY_QUEUED,
    EMAIL_DELIVERY_SENDING,
    EMAIL_DELIVERY_SENT,
    ActivationSession,
)
from services.notifications.email import is_email_provider_configured

logger = logging.getLogger(__name__)

REDIS_TOKEN_KEY_PREFIX = "activation:email_token:"
ACTIVATION_EMAIL_TTL_MINUTES = int(os.getenv("ACTIVATION_EMAIL_TTL_MINUTES", "30"))


def activation_token_redis_key(email_delivery_id: str) -> str:
    return f"{REDIS_TOKEN_KEY_PREFIX}{email_delivery_id}"


def sanitize_email_error(message: str | None) -> str:
    """Supprime jetons, URLs et emails avant stockage interne."""
    if not message:
        return "email_send_failed"
    text = str(message)
    text = re.sub(r"https?://\S+", "[url]", text)
    text = re.sub(r"[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-.]+", "[token]", text)
    text = re.sub(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        "[email]",
        text,
    )
    return text[:500]


def store_activation_email_token(email_delivery_id: str, token: str) -> None:
    """Stocke le jeton clair en Redis (TTL = expiration email)."""
    if redis_client is None:
        raise RuntimeError("Redis unavailable")
    ttl_seconds = max(60, ACTIVATION_EMAIL_TTL_MINUTES * 60)
    key = activation_token_redis_key(email_delivery_id)
    redis_client.setex(key, ttl_seconds, token)


def get_activation_email_token(email_delivery_id: str) -> str | None:
    if redis_client is None:
        return None
    raw = redis_client.get(activation_token_redis_key(email_delivery_id))
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def purge_activation_email_token(email_delivery_id: str) -> None:
    if redis_client is None:
        return
    try:
        redis_client.delete(activation_token_redis_key(email_delivery_id))
    except Exception:
        logger.warning(
            "[activation_email] purge Redis échouée delivery_id=%s",
            email_delivery_id,
        )


def mark_delivery_failed(
    session: ActivationSession,
    error: str,
    *,
    email_delivery_id: str | None = None,
) -> None:
    """Marque l'envoi courant en failed (si delivery_id correspond ou None)."""
    if (
        email_delivery_id
        and session.email_delivery_id
        and session.email_delivery_id != email_delivery_id
    ):
        return
    session.email_delivery_status = EMAIL_DELIVERY_FAILED
    session.email_last_error = sanitize_email_error(error)


def prepare_activation_email_delivery(
    session: ActivationSession,
    *,
    kind: str,
    email_token: str,
    email_token_hash: str,
    token_expires_at: datetime,
) -> str:
    """Prépare un nouvel envoi : delivery_id, hash, statut queued.

    Ne touche pas last_email_sent_at / resend_count (après succès Brevo uniquement).
    """
    if kind not in {EMAIL_DELIVERY_KIND_INITIAL, EMAIL_DELIVERY_KIND_RESEND}:
        raise ValueError(f"kind invalide: {kind}")

    delivery_id = str(uuid.uuid4())
    session.email_delivery_id = delivery_id
    session.email_delivery_status = EMAIL_DELIVERY_QUEUED
    session.email_delivery_kind = kind
    session.email_token_hash = email_token_hash
    session.email_token_expires_at = token_expires_at
    session.email_last_error = None
    session.email_provider_message_id = None
    store_activation_email_token(delivery_id, email_token)
    return delivery_id


def enqueue_activation_email(
    *,
    activation_session_id: str,
    email_delivery_id: str,
) -> None:
    """Enqueue Celery (args non sensibles uniquement)."""
    from tasks.notification_tasks import send_activation_email_task

    send_activation_email_task.delay(
        activation_session_id=activation_session_id,
        email_delivery_id=email_delivery_id,
    )


def try_enqueue_activation_email(
    session: ActivationSession,
    *,
    kind: str,
    email_token: str,
    email_token_hash: str,
    environment: str,
    is_testing: bool,
) -> dict[str, Any]:
    """Prépare + enqueue un envoi d'activation avec compensations.

    Returns:
        dict avec keys:
          ok (bool), queued (bool), email_sent (None),
          debug_activation_link (str|None), error (str|None),
          require_502 (bool)
    """
    now = datetime.now(UTC)
    token_expires_at = now + timedelta(minutes=ACTIVATION_EMAIL_TTL_MINUTES)
    env = (environment or "").strip().lower()
    is_prod = env == "production"
    is_local_dev = env == "development" and not is_testing

    ready, config_error = is_email_provider_configured()
    if not ready and is_prod:
        mark_delivery_failed(session, config_error or "email_provider_not_configured")
        # delivery_id peut être absent : on force un statut failed cohérent
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
        }

    delivery_id: str | None = None
    try:
        delivery_id = prepare_activation_email_delivery(
            session,
            kind=kind,
            email_token=email_token,
            email_token_hash=email_token_hash,
            token_expires_at=token_expires_at,
        )
        db.session.commit()
    except Exception as e:
        logger.exception("[activation_email] échec préparation Redis/DB: %s", e)
        try:
            mark_delivery_failed(session, str(e))
            db.session.commit()
        except Exception:
            db.session.rollback()
        debug_link = None
        if is_local_dev:
            # Lien de secours local uniquement (jeton déjà généré côté appelant).
            frontend = (
                os.getenv("FRONTEND_URL")
                or os.getenv("PUBLIC_FRONTEND_URL")
                or "http://localhost:3000"
            ).rstrip("/")
            debug_link = f"{frontend}/activate-account?token={email_token}"
        return {
            "ok": is_local_dev,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": debug_link,
            "error": sanitize_email_error(str(e)),
            "require_502": not is_local_dev and not is_testing,
        }

    if not ready and is_local_dev:
        # Dev sans Brevo : jeton en Redis, pas d'enqueue, lien de secours.
        frontend = (
            os.getenv("FRONTEND_URL")
            or os.getenv("PUBLIC_FRONTEND_URL")
            or "http://localhost:3000"
        ).rstrip("/")
        return {
            "ok": True,
            "queued": False,
            "email_sent": None,
            "debug_activation_link": f"{frontend}/activate-account?token={email_token}",
            "error": None,
            "require_502": False,
        }

    try:
        enqueue_activation_email(
            activation_session_id=session.activation_session_id,
            email_delivery_id=delivery_id,
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
        }

    return {
        "ok": True,
        "queued": True,
        "email_sent": None,
        "debug_activation_link": None,
        "error": None,
        "require_502": False,
    }


def cas_claim_sending(
    session: ActivationSession,
    email_delivery_id: str,
) -> str:
    """Claim atomique pour la tâche Celery.

    Returns:
        'proceed' | 'ignore' | 'missing_token'
    """
    if session.email_verified_at is not None:
        return "ignore"
    if session.email_delivery_id != email_delivery_id:
        return "ignore"
    if session.email_delivery_status == EMAIL_DELIVERY_SENT:
        return "ignore"

    if session.email_delivery_status == EMAIL_DELIVERY_QUEUED:
        # CAS SQL
        updated = (
            ActivationSession.query.filter_by(
                id=session.id,
                email_delivery_id=email_delivery_id,
                email_delivery_status=EMAIL_DELIVERY_QUEUED,
            )
            .update(
                {"email_delivery_status": EMAIL_DELIVERY_SENDING},
                synchronize_session=False,
            )
        )
        db.session.commit()
        if updated == 0:
            db.session.refresh(session)
            if (
                session.email_delivery_id == email_delivery_id
                and session.email_delivery_status == EMAIL_DELIVERY_SENDING
            ):
                return "proceed"
            return "ignore"
        db.session.refresh(session)
        return "proceed"

    if (
        session.email_delivery_status == EMAIL_DELIVERY_SENDING
        and session.email_delivery_id == email_delivery_id
    ):
        # Retry du même delivery_id
        return "proceed"

    return "ignore"


def mark_delivery_sent(
    session: ActivationSession,
    *,
    email_delivery_id: str,
    message_id: str | None,
) -> None:
    if session.email_delivery_id != email_delivery_id:
        return
    now = datetime.now(UTC)
    previous_sent_at = session.last_email_sent_at
    if session.email_delivery_kind == EMAIL_DELIVERY_KIND_RESEND:

        def _same_day(a: datetime, b: datetime) -> bool:
            return a.astimezone(UTC).date() == b.astimezone(UTC).date()

        if previous_sent_at and not _same_day(previous_sent_at, now):
            session.resend_count_email = 1
        else:
            session.resend_count_email = int(session.resend_count_email or 0) + 1
    session.email_delivery_status = EMAIL_DELIVERY_SENT
    session.last_email_sent_at = now
    session.email_provider_message_id = message_id
    session.email_last_error = None
    purge_activation_email_token(email_delivery_id)
