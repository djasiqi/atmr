"""Couverture push chauffeur pour l'endpoint admin."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func

from ext import redis_client
from models import DeviceToken, Driver
from services.notifications.push_device_selection import (
    android_has_expo_only,
)
from services.notifications.push_token_classification import classify_token

INVALID_TOKEN_CODES = frozenset({"token_unregistered", "DeviceNotRegistered"})


def _max_push_success_at(tokens: list[DeviceToken]) -> datetime | None:
    best: datetime | None = None
    for token in tokens:
        if token.last_push_success_at is None:
            continue
        if best is None or token.last_push_success_at > best:
            best = token.last_push_success_at
    return best


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat()


def _read_device_health(driver_id: int) -> dict[str, Any]:
    if redis_client is None:
        return {}
    try:
        raw = redis_client.get(f"driver:{driver_id}:device_health")
        if not raw:
            raw = redis_client.hgetall(f"driver:{driver_id}:device_health")
            if isinstance(raw, dict) and raw:
                decoded: dict[str, Any] = {}
                for key, val in raw.items():
                    k = key.decode() if isinstance(key, bytes) else str(key)
                    v = val.decode() if isinstance(val, bytes) else val
                    decoded[k] = v
                return decoded
            return {}
        if isinstance(raw, bytes):
            raw = raw.decode()
        return json.loads(raw) if isinstance(raw, str) else {}
    except Exception:
        return {}


def _read_driver_last_seen(driver_id: int) -> datetime | None:
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(f"driver:{driver_id}:last_seen")
        if not raw:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode()
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None


def _resolve_push_status(
    *,
    has_active_token: bool,
    active_tokens: list[DeviceToken],
) -> str:
    if not has_active_token:
        return "no_token"
    for token in active_tokens:
        if (token.last_push_error_code or "") in INVALID_TOKEN_CODES:
            return "token_invalid"
    if android_has_expo_only(active_tokens):
        return "expo_fallback_unreliable"
    classifications = {classify_token(token) for token in active_tokens}
    if classifications & {"STALE", "MISMATCH_PROVIDER"}:
        return "stale_token"
    return "operational"


def build_driver_push_coverage_row(driver: Driver) -> dict[str, Any]:
    active_tokens = (
        DeviceToken.query.filter_by(driver_id=driver.id, is_active=True)
        .order_by(DeviceToken.updated_at.desc())
        .all()
    )
    latest_token = active_tokens[0] if active_tokens else None
    health = _read_device_health(int(driver.id))

    last_seen_db = (
        DeviceToken.query.with_entities(func.max(DeviceToken.last_seen_at))
        .filter(DeviceToken.driver_id == driver.id)
        .scalar()
    )
    redis_seen = _read_driver_last_seen(int(driver.id))
    last_driver_activity_at = last_seen_db
    if redis_seen and (
        last_driver_activity_at is None or redis_seen > last_driver_activity_at
    ):
        last_driver_activity_at = redis_seen

    push_operational = bool(driver.is_active and driver.is_available)
    has_active_token = len(active_tokens) > 0
    push_status = _resolve_push_status(
        has_active_token=has_active_token,
        active_tokens=active_tokens,
    )

    return {
        "driver_id": driver.id,
        "company_id": driver.company_id,
        "is_active": bool(driver.is_active),
        "is_available": bool(driver.is_available),
        "push_operational": push_operational,
        "has_active_token": has_active_token,
        "active_tokens_count": len(active_tokens),
        "last_driver_activity_at": _iso(last_driver_activity_at),
        "token_created_at": _iso(latest_token.created_at if latest_token else None),
        "token_updated_at": _iso(latest_token.updated_at if latest_token else None),
        "last_push_success_at": _iso(
            _max_push_success_at(active_tokens) if active_tokens else None
        ),
        "last_push_error_code": latest_token.last_push_error_code
        if latest_token
        else None,
        "platform": latest_token.platform if latest_token else None,
        "provider": latest_token.provider if latest_token else None,
        "app_version": health.get("app_version") or "",
        "token_classifications": [classify_token(t) for t in active_tokens],
        "push_status": push_status,
    }


def list_driver_push_coverage(
    *,
    company_id: int | None = None,
    driver_id: int | None = None,
    without_token_only: bool = False,
    operational_only: bool = True,
) -> list[dict[str, Any]]:
    query = Driver.query
    if company_id is not None:
        query = query.filter(Driver.company_id == company_id)
    if driver_id is not None:
        query = query.filter(Driver.id == driver_id)
    if operational_only:
        query = query.filter(Driver.is_active.is_(True), Driver.is_available.is_(True))

    drivers = query.order_by(Driver.id.asc()).all()
    rows = [build_driver_push_coverage_row(driver) for driver in drivers]
    if without_token_only:
        rows = [row for row in rows if not row["has_active_token"]]
    return rows
