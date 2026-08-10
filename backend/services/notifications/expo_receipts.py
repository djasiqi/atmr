"""Suivi des tickets Expo Push et application des receipts (Phase A)."""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from ext import app_logger, redis_client
from services.notifications.push_delivery_status import (
    RECEIPT_PENDING,
    apply_expo_receipt_to_classification,
    classify_push_result,
    log_push_attempt_event,
)

EXPO_RECEIPT_REDIS_PREFIX = "push:expo_ticket:"
EXPO_RECEIPT_TTL_SEC = int(os.getenv("PUSH_EXPO_TICKET_TTL_SEC", str(90 * 24 * 3600)))
EXPO_GET_RECEIPTS_URL = "https://exp.host/--/api/v2/push/getReceipts"


def store_expo_ticket(
    *,
    ticket_id: str,
    correlation_id: str | None,
    device_token_id: int | None,
    driver_id: int | None,
    platform: str | None,
    notification_type: str | None,
    deduplication_key: str | None = None,
) -> None:
    """Persiste la corrélation ticket ↔ tentative (Redis, TTL 90j preuves)."""
    if not ticket_id or not redis_client:
        return
    payload = {
        "ticket_id": ticket_id,
        "correlation_id": correlation_id,
        "device_token_id": device_token_id,
        "driver_id": driver_id,
        "platform": platform,
        "notification_type": notification_type,
        "deduplication_key": deduplication_key,
        "provider_receipt_status": RECEIPT_PENDING,
    }
    try:
        redis_client.setex(
            f"{EXPO_RECEIPT_REDIS_PREFIX}{ticket_id}",
            EXPO_RECEIPT_TTL_SEC,
            json.dumps(payload, default=str),
        )
    except Exception as e:
        app_logger.warning("[expo_receipts] store ticket failed: %s", str(e)[:200])


def load_expo_ticket(ticket_id: str) -> dict[str, Any] | None:
    if not ticket_id or not redis_client:
        return None
    try:
        raw = redis_client.get(f"{EXPO_RECEIPT_REDIS_PREFIX}{ticket_id}")
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        return json.loads(raw)
    except Exception:
        return None


def fetch_expo_receipts(ticket_ids: list[str]) -> dict[str, Any]:
    """Appelle l'API Expo getReceipts. Retourne le dict ``data`` (ticket_id → receipt)."""
    if not ticket_ids:
        return {}
    resp = requests.post(
        EXPO_GET_RECEIPTS_URL,
        json={"ids": ticket_ids},
        timeout=10,
    )
    resp.raise_for_status()
    body = resp.json()
    return body.get("data") or {}


def apply_expo_receipts(
    ticket_ids: list[str] | None = None,
    *,
    max_tickets: int = 100,
) -> dict[str, Any]:
    """Récupère et applique les receipts Expo (invalidation DeviceNotRegistered seulement)."""
    from services.notifications.device_token_lifecycle import (
        apply_push_result_to_device_token,
    )
    from ext import db

    ids = list(ticket_ids or [])
    if not ids and redis_client:
        try:
            # Scan limité des tickets pending
            cursor = 0
            pattern = f"{EXPO_RECEIPT_REDIS_PREFIX}*"
            while len(ids) < max_tickets:
                cursor, keys = redis_client.scan(cursor, match=pattern, count=50)
                for key in keys:
                    key_s = key.decode() if isinstance(key, bytes) else str(key)
                    tid = key_s.replace(EXPO_RECEIPT_REDIS_PREFIX, "", 1)
                    meta = load_expo_ticket(tid) or {}
                    if meta.get("provider_receipt_status") == RECEIPT_PENDING:
                        ids.append(tid)
                    if len(ids) >= max_tickets:
                        break
                if cursor == 0:
                    break
        except Exception as e:
            app_logger.warning("[expo_receipts] scan failed: %s", str(e)[:200])

    if not ids:
        return {"processed": 0, "updated": 0, "pending": 0}

    try:
        receipts = fetch_expo_receipts(ids)
    except requests.RequestException as e:
        app_logger.warning("[expo_receipts] fetch failed: %s", str(e)[:200])
        return {"processed": 0, "updated": 0, "error": str(e)[:200]}

    updated = 0
    still_pending = 0
    for tid in ids:
        rec = receipts.get(tid)
        meta = load_expo_ticket(tid) or {"ticket_id": tid}
        if rec is None:
            still_pending += 1
            continue

        status = rec.get("status", "error")
        details = rec.get("details") or {}
        err = details.get("error") if isinstance(details, dict) else None

        base = classify_push_result(
            {"ok": True, "provider_ticket_id": tid},
            provider="expo",
        )
        classified = apply_expo_receipt_to_classification(
            base,
            receipt_status=str(status),
            receipt_error=str(err) if err else None,
        )

        log_push_attempt_event(
            delivery_status=classified["delivery_status"],
            platform=meta.get("platform"),
            provider="expo",
            device_token_id=meta.get("device_token_id"),
            notification_type=meta.get("notification_type"),
            correlation_id=meta.get("correlation_id"),
            driver_id=meta.get("driver_id"),
            provider_receipt_status=classified["provider_receipt_status"],
            failure_reason=classified.get("failure_reason"),
            provider_ticket_id=tid,
            provider_error_code=classified.get("provider_error_code"),
            provider_error_category=classified.get("provider_error_category"),
            deduplication_key=meta.get("deduplication_key"),
            extra={"event": "push_attempt_receipt"},
        )

        device_token_id = meta.get("device_token_id")
        if device_token_id is not None:
            apply_push_result_to_device_token(
                int(device_token_id),
                {
                    "ok": classified["delivery_status"] in ("provider_accepted",),
                    "error": classified.get("failure_reason"),
                    "token_invalid": classified.get("deactivate_token"),
                    "deactivate_token": classified.get("deactivate_token"),
                    "configuration_error": classified.get("delivery_status")
                    == "configuration_error",
                    "delivery_status": classified.get("delivery_status"),
                },
            )
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()

        meta["provider_receipt_status"] = classified["provider_receipt_status"]
        meta["delivery_status"] = classified["delivery_status"]
        if redis_client:
            try:
                redis_client.setex(
                    f"{EXPO_RECEIPT_REDIS_PREFIX}{tid}",
                    EXPO_RECEIPT_TTL_SEC,
                    json.dumps(meta, default=str),
                )
            except Exception:
                pass
        updated += 1

    return {
        "processed": len(ids),
        "updated": updated,
        "pending": still_pending,
    }
