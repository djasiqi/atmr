"""Webhooks Brevo transactionnels — Bearer + idempotence atomique (Lot 1)."""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Any

from sqlalchemy import text

from ext import db
from models.activation_email_delivery import (
    EMAIL_DELIVERY_BLOCKED,
    EMAIL_DELIVERY_DELIVERED,
    EMAIL_DELIVERY_HARD_BOUNCED,
    EMAIL_DELIVERY_INVALID,
    EMAIL_DELIVERY_SOFT_BOUNCED,
    EMAIL_DELIVERY_SPAM,
    ActivationEmailDelivery,
)
from models.activation_session import ActivationSession
from services.notifications.activation_email_delivery import (
    apply_delivery_transition,
    get_delivery_by_id,
)

logger = logging.getLogger(__name__)

# Mapping event Brevo → statut livraison
_BREVO_EVENT_TO_STATUS: dict[str, str] = {
    "delivered": EMAIL_DELIVERY_DELIVERED,
    "soft_bounce": EMAIL_DELIVERY_SOFT_BOUNCED,
    "hard_bounce": EMAIL_DELIVERY_HARD_BOUNCED,
    "blocked": EMAIL_DELIVERY_BLOCKED,
    "invalid_email": EMAIL_DELIVERY_INVALID,
    "spam": EMAIL_DELIVERY_SPAM,
    "complaint": EMAIL_DELIVERY_SPAM,
}


def brevo_webhook_secret() -> str:
    return (os.getenv("BREVO_WEBHOOK_SECRET") or "").strip()


def require_brevo_webhook_secret_in_production() -> None:
    env = (os.getenv("ENVIRONMENT") or "").strip().lower()
    if env == "production" and not brevo_webhook_secret():
        raise RuntimeError("BREVO_WEBHOOK_SECRET obligatoire en production")


def verify_brevo_bearer(authorization_header: str | None) -> bool:
    """Valide Authorization: Bearer <secret> en temps constant."""
    expected = brevo_webhook_secret()
    env = (os.getenv("ENVIRONMENT") or "").strip().lower()
    if not expected:
        if env == "production":
            return False
        # Hors prod : refuser si secret absent (tests doivent le définir)
        return False
    if not authorization_header:
        return False
    parts = authorization_header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return False
    provided = parts[1].strip()
    return hmac.compare_digest(provided, expected)


def compute_idempotency_key(
    *,
    message_id: str,
    event: str,
    ts_event: str,
    email: str,
) -> str:
    raw = (
        f"{str(message_id).strip()}|"
        f"{str(event).strip()}|"
        f"{str(ts_event).strip()}|"
        f"{str(email).strip().lower()}"
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _extract_custom_delivery_id(payload: dict[str, Any]) -> str | None:
    """X-Mailin-custom / champs Brevo équivalents."""
    for key in (
        "X-Mailin-custom",
        "x-mailin-custom",
        "mailin-custom",
        "custom",
    ):
        val = payload.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    headers = payload.get("headers")
    if isinstance(headers, dict):
        for key in ("X-Mailin-custom", "x-mailin-custom"):
            val = headers.get(key)
            if val is not None and str(val).strip():
                return str(val).strip()
    return None


def resolve_delivery_from_webhook(
    payload: dict[str, Any],
) -> ActivationEmailDelivery | None:
    custom_id = _extract_custom_delivery_id(payload)
    if custom_id:
        delivery = get_delivery_by_id(custom_id)
        if delivery:
            return delivery
    message_id = str(
        payload.get("message-id") or payload.get("messageId") or ""
    ).strip()
    if message_id:
        return ActivationEmailDelivery.query.filter_by(
            provider_message_id=message_id
        ).first()
    return None


def process_brevo_webhook_event(payload: dict[str, Any]) -> dict[str, Any]:
    """Insert event + transition dans une seule transaction.

    Returns:
        dict with keys: status ('ok'|'noop'|'ignored'), http_status
    """
    event = str(payload.get("event") or "").strip().lower()
    message_id = str(
        payload.get("message-id") or payload.get("messageId") or ""
    ).strip()
    ts_event = str(payload.get("ts_event") or payload.get("date") or "").strip()
    email = str(payload.get("email") or "").strip().lower()

    if not event or not message_id:
        return {"status": "ignored", "http_status": 200, "reason": "missing_fields"}

    new_status = _BREVO_EVENT_TO_STATUS.get(event)
    if not new_status:
        return {"status": "ignored", "http_status": 200, "reason": "unhandled_event"}

    idem_key = compute_idempotency_key(
        message_id=message_id,
        event=event,
        ts_event=ts_event,
        email=email,
    )

    delivery = resolve_delivery_from_webhook(payload)
    delivery_id = delivery.email_delivery_id if delivery else None

    try:
        # ON CONFLICT DO NOTHING RETURNING — évite abort txn PostgreSQL
        result = db.session.execute(
            text(
                """
                INSERT INTO brevo_webhook_events
                  (idempotency_key, event_type, provider_message_id, email_delivery_id)
                VALUES
                  (:ikey, :etype, :mid, :did)
                ON CONFLICT (idempotency_key) DO NOTHING
                RETURNING id
                """
            ),
            {
                "ikey": idem_key,
                "etype": event,
                "mid": message_id or None,
                "did": delivery_id,
            },
        )
        row = result.fetchone()
        if row is None:
            db.session.rollback()
            return {"status": "noop", "http_status": 200, "reason": "duplicate"}

        if delivery is None:
            # Événement enregistré mais livraison inconnue — commit event seul
            db.session.commit()
            return {
                "status": "ignored",
                "http_status": 200,
                "reason": "unknown_delivery",
            }

        applied = apply_delivery_transition(delivery, new_status)
        if applied and not delivery.provider_message_id and message_id:
            delivery.provider_message_id = message_id

        # Sync session courante uniquement si c'est la livraison active
        session = ActivationSession.query.get(delivery.activation_session_pk)
        if (
            session
            and session.email_delivery_id == delivery.email_delivery_id
            and applied
        ):
            session.email_delivery_status = delivery.status
            session.email_provider_message_id = delivery.provider_message_id
            if new_status in {
                EMAIL_DELIVERY_HARD_BOUNCED,
                EMAIL_DELIVERY_SPAM,
                EMAIL_DELIVERY_BLOCKED,
                EMAIL_DELIVERY_INVALID,
            }:
                session.email_last_error = f"brevo_{event}"

        db.session.commit()
        return {
            "status": "ok",
            "http_status": 200,
            "applied": applied,
            "delivery_id": delivery.email_delivery_id,
        }
    except Exception:
        db.session.rollback()
        logger.exception("[brevo_webhook] échec traitement atomique")
        raise


__all__ = [
    "compute_idempotency_key",
    "process_brevo_webhook_event",
    "require_brevo_webhook_secret_in_production",
    "verify_brevo_bearer",
]
