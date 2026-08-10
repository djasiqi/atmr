"""Statuts canoniques de livraison push (Phase A — observabilité iOS/FCM/Expo).

Une seule valeur canonique est stockée / journalisée. Les alias ``sent`` /
``rejected`` / ``delivered`` existent uniquement pour traduction dashboard.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

# --- Statuts canoniques (stockage / logs) -------------------------------------

QUEUED = "queued"
PROVIDER_ACCEPTED = "provider_accepted"
PROVIDER_REJECTED = "provider_rejected"
CONFIGURATION_ERROR = "configuration_error"
INVALID_TOKEN = "invalid_token"
RETRY_PENDING = "retry_pending"
FAILED = "failed"
MOBILE_RECEIVED = "mobile_received"
MOBILE_OPENED = "mobile_opened"
BUSINESS_ACKNOWLEDGED = "business_acknowledged"

CANONICAL_DELIVERY_STATUSES = frozenset(
    {
        QUEUED,
        PROVIDER_ACCEPTED,
        PROVIDER_REJECTED,
        CONFIGURATION_ERROR,
        INVALID_TOKEN,
        RETRY_PENDING,
        FAILED,
        MOBILE_RECEIVED,
        MOBILE_OPENED,
        BUSINESS_ACKNOWLEDGED,
    }
)

# Receipt Expo / cycle provider secondaire
RECEIPT_PENDING = "pending"
RECEIPT_OK = "ok"
RECEIPT_ERROR = "error"
RECEIPT_NOT_APPLICABLE = "not_applicable"

# Alias dashboards uniquement (ne jamais écrire en nouveau stockage)
DASHBOARD_STATUS_ALIASES = {
    "sent": PROVIDER_ACCEPTED,
    "rejected": PROVIDER_REJECTED,
    "delivered": MOBILE_RECEIVED,
}

PROVIDER_RESPONSE_SANITIZED_MAX_LEN = int(
    os.getenv("PUSH_PROVIDER_RESPONSE_SANITIZED_MAX_LEN", "256")
)

# Rétention documentée (ops) — pas d'application automatique ici
LOG_RETENTION_DAYS = 30
TEST_PUSH_PROOF_RETENTION_DAYS = 90

_TOKEN_LIKE = re.compile(
    r"(ExponentPushToken\[[^\]]+\]|[A-Za-z0-9_-]{80,})",
    re.IGNORECASE,
)


def canonicalize_delivery_status(status: str | None) -> str | None:
    """Normalise un statut (alias dashboard → canonique)."""
    if status is None:
        return None
    s = str(status).strip().lower()
    if s in CANONICAL_DELIVERY_STATUSES:
        return s
    return DASHBOARD_STATUS_ALIASES.get(s)


def sanitize_provider_text(value: Any, *, max_len: int | None = None) -> str:
    """Expurge tokens / payloads longs pour logs et stockage."""
    limit = max_len if max_len is not None else PROVIDER_RESPONSE_SANITIZED_MAX_LEN
    text = "" if value is None else str(value)
    text = _TOKEN_LIKE.sub("[REDACTED]", text)
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def classify_push_result(
    result: dict[str, Any] | None,
    *,
    provider: str | None = None,
) -> dict[str, Any]:
    """Mappe un résultat d'envoi push vers statuts / champs provider normalisés.

    Returns:
        dict avec delivery_status, provider_receipt_status, failure_reason,
        provider_message_id, provider_ticket_id, provider_error_code,
        provider_error_category, provider_http_status, provider_response_sanitized,
        token_invalid (True uniquement si invalid_token confirmé).
    """
    result = result or {}
    provider_l = (provider or "").lower()

    out: dict[str, Any] = {
        "delivery_status": FAILED,
        "provider_receipt_status": RECEIPT_NOT_APPLICABLE,
        "failure_reason": None,
        "provider_message_id": None,
        "provider_ticket_id": None,
        "provider_error_code": None,
        "provider_error_category": None,
        "provider_http_status": result.get("http_status"),
        "provider_response_sanitized": None,
        "token_invalid": False,
        "deactivate_token": False,
    }

    error = result.get("error")
    error_code = str(error or result.get("error_class") or "")
    error_l = error_code.lower()

    # Circuit breaker / rate limit → retryable côté appelant
    if result.get("circuit_breaker_open") or error == "circuit_breaker_open":
        out["delivery_status"] = RETRY_PENDING
        out["failure_reason"] = "circuit_breaker_open"
        out["provider_error_code"] = "circuit_breaker_open"
        out["provider_error_category"] = "retryable"
        return out

    if result.get("rate_limit_exceeded"):
        out["delivery_status"] = RETRY_PENDING
        out["failure_reason"] = "rate_limit_exceeded"
        out["provider_error_code"] = "rate_limit_exceeded"
        out["provider_error_category"] = "retryable"
        return out

    if result.get("retry_exhausted") or error == "retry_exhausted":
        out["delivery_status"] = FAILED
        out["failure_reason"] = "retry_exhausted"
        out["provider_error_code"] = error_code or "retry_exhausted"
        out["provider_error_category"] = "exhausted"
        out["provider_response_sanitized"] = sanitize_provider_text(
            result.get("error_message") or error
        )
        return out

    # Configuration — ne jamais désactiver le token
    if (
        error == "sender_id_mismatch"
        or "senderidmismatch" in error_l
        or "sender_id_mismatch" in error_l
        or error == "InvalidCredentials"
        or result.get("configuration_error")
    ):
        out["delivery_status"] = CONFIGURATION_ERROR
        out["failure_reason"] = error_code or "configuration_error"
        out["provider_error_code"] = error_code or "configuration_error"
        out["provider_error_category"] = "configuration"
        out["provider_response_sanitized"] = sanitize_provider_text(
            result.get("error_message") or error
        )
        out["token_invalid"] = False
        out["deactivate_token"] = False
        return out

    # Token invalide confirmé
    if (
        result.get("token_invalid")
        or error in ("token_unregistered", "DeviceNotRegistered")
        or "unregistered" in error_l
        or "devicenotregistered" in error_l
    ):
        # InvalidCredentials déjà traité comme configuration ci-dessus
        if error == "InvalidCredentials":
            out["delivery_status"] = CONFIGURATION_ERROR
            out["provider_error_category"] = "configuration"
            out["deactivate_token"] = False
            return out
        out["delivery_status"] = INVALID_TOKEN
        out["failure_reason"] = error_code or "invalid_token"
        out["provider_error_code"] = error_code or "invalid_token"
        out["provider_error_category"] = "invalid_token"
        out["token_invalid"] = True
        out["deactivate_token"] = True
        out["provider_response_sanitized"] = sanitize_provider_text(
            result.get("error_message") or error
        )
        return out

    if result.get("ok"):
        message_id = result.get("message_id")
        ticket_id = result.get("provider_ticket_id") or result.get("expo_ticket_id")
        if not ticket_id and isinstance(result.get("data"), list):
            for ticket in result["data"]:
                if isinstance(ticket, dict) and ticket.get("id"):
                    ticket_id = ticket.get("id")
                    break

        out["delivery_status"] = PROVIDER_ACCEPTED
        out["provider_message_id"] = str(message_id) if message_id else None
        out["provider_ticket_id"] = str(ticket_id) if ticket_id else None
        out["failure_reason"] = None
        out["provider_error_category"] = None

        if provider_l == "expo" and ticket_id:
            out["provider_receipt_status"] = RECEIPT_PENDING
        elif provider_l == "fcm":
            out["provider_receipt_status"] = RECEIPT_NOT_APPLICABLE
        else:
            out["provider_receipt_status"] = (
                RECEIPT_PENDING if ticket_id else RECEIPT_NOT_APPLICABLE
            )
        return out

    # Erreurs retryables
    retryable_hints = (
        "timeout",
        "timed out",
        "unavailable",
        "connection",
        "503",
        "500",
        "429",
        "network",
    )
    msg = str(result.get("error_message") or error or "").lower()
    if result.get("retryable") or any(h in msg for h in retryable_hints):
        out["delivery_status"] = RETRY_PENDING
        out["failure_reason"] = error_code or "retryable_error"
        out["provider_error_code"] = error_code or "retryable_error"
        out["provider_error_category"] = "retryable"
        out["provider_response_sanitized"] = sanitize_provider_text(
            result.get("error_message") or error
        )
        return out

    # Refus provider / erreur définitive
    out["delivery_status"] = PROVIDER_REJECTED
    out["failure_reason"] = error_code or "provider_rejected"
    out["provider_error_code"] = error_code or "provider_rejected"
    out["provider_error_category"] = "rejected"
    out["provider_response_sanitized"] = sanitize_provider_text(
        result.get("error_message") or error
    )
    return out


def apply_expo_receipt_to_classification(
    classification: dict[str, Any],
    *,
    receipt_status: str,
    receipt_error: str | None = None,
) -> dict[str, Any]:
    """Met à jour une classification après receipt Expo."""
    out = dict(classification)
    status_l = (receipt_status or "").lower()
    err = receipt_error or ""

    if status_l == "ok":
        out["provider_receipt_status"] = RECEIPT_OK
        out["delivery_status"] = PROVIDER_ACCEPTED
        out["failure_reason"] = None
        out["token_invalid"] = False
        out["deactivate_token"] = False
        return out

    out["provider_receipt_status"] = RECEIPT_ERROR
    out["provider_error_code"] = err or "expo_receipt_error"
    out["provider_response_sanitized"] = sanitize_provider_text(err)

    if err in ("DeviceNotRegistered",) or "devicenotregistered" in err.lower():
        out["delivery_status"] = INVALID_TOKEN
        out["failure_reason"] = "DeviceNotRegistered"
        out["provider_error_category"] = "invalid_token"
        out["token_invalid"] = True
        out["deactivate_token"] = True
    elif err in ("InvalidCredentials",):
        out["delivery_status"] = CONFIGURATION_ERROR
        out["failure_reason"] = "InvalidCredentials"
        out["provider_error_category"] = "configuration"
        out["token_invalid"] = False
        out["deactivate_token"] = False
    else:
        out["delivery_status"] = PROVIDER_REJECTED
        out["failure_reason"] = err or "expo_receipt_error"
        out["provider_error_category"] = "rejected"
        out["token_invalid"] = False
        out["deactivate_token"] = False
    return out


def log_push_attempt_event(
    *,
    delivery_status: str,
    platform: str | None = None,
    provider: str | None = None,
    device_token_id: int | None = None,
    notification_type: str | None = None,
    correlation_id: str | None = None,
    driver_id: int | None = None,
    provider_receipt_status: str | None = None,
    failure_reason: str | None = None,
    provider_message_id: str | None = None,
    provider_ticket_id: str | None = None,
    provider_error_code: str | None = None,
    provider_error_category: str | None = None,
    provider_http_status: int | str | None = None,
    provider_response_sanitized: str | None = None,
    deduplication_key: str | None = None,
    notification_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Journalise une tentative push structurée (sans données sensibles)."""
    canonical = canonicalize_delivery_status(delivery_status) or delivery_status
    payload: dict[str, Any] = {
        "event": "push_attempt",
        "delivery_status": canonical,
        "platform": platform,
        "provider": provider,
        "device_token_id": device_token_id,
        "notification_type": notification_type,
        "correlation_id": correlation_id,
        "driver_id": driver_id,
        "provider_receipt_status": provider_receipt_status,
        "failure_reason": failure_reason,
        "provider_message_id": provider_message_id,
        "provider_ticket_id": provider_ticket_id,
        "provider_error_code": provider_error_code,
        "provider_error_category": provider_error_category,
        "provider_http_status": provider_http_status,
        "provider_response_sanitized": provider_response_sanitized,
        "deduplication_key": deduplication_key,
        "notification_id": notification_id,
        "retention_hint_days": LOG_RETENTION_DAYS,
    }
    if extra:
        for k, v in extra.items():
            if k not in payload:
                payload[k] = v

    # Retirer les None pour alléger les logs
    compact = {k: v for k, v in payload.items() if v is not None}
    logger.info("[push_attempt] %s", json.dumps(compact, default=str))

    try:
        from services.notifications.metrics import record_push_attempt_status

        record_push_attempt_status(
            platform=platform or "unknown",
            provider=provider or "unknown",
            delivery_status=canonical,
            notification_type=notification_type or "unknown",
            error_category=provider_error_category or "none",
        )
    except Exception:
        pass


def ensure_deduplication_fields(
    data: dict[str, Any] | None,
    *,
    notification_type: str | None = None,
    driver_id: int | None = None,
) -> dict[str, Any]:
    """Garantit notification_id + deduplication_key dans le payload data."""
    import uuid

    out = dict(data or {})
    if not out.get("notification_id"):
        out["notification_id"] = str(
            out.get("event_id") or out.get("trace_id") or uuid.uuid4()
        )

    # Alias historique → clé canonique
    if out.get("dedupe_key") and not out.get("deduplication_key"):
        out["deduplication_key"] = out["dedupe_key"]
    if out.get("deduplication_key") and not out.get("dedupe_key"):
        out["dedupe_key"] = out["deduplication_key"]

    if not out.get("deduplication_key"):
        ntype = notification_type or out.get("type") or "push"
        booking = out.get("booking_id") or out.get("mission_id") or "na"
        driver = driver_id if driver_id is not None else out.get("driver_id") or "na"
        out["deduplication_key"] = (
            f"{ntype}:{booking}:{driver}:{out['notification_id']}"
        )
        out["dedupe_key"] = out["deduplication_key"]

    return out
