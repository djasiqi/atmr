#!/usr/bin/env python
"""Audit des tokens push chauffeur (device_tokens).

Classe chaque token : HEALTHY, STALE, INACTIVE, MISMATCH_PROVIDER.

Usage (via Docker) :
    docker compose exec api python scripts/audit_device_tokens.py --report
    docker compose exec api python scripts/audit_device_tokens.py --list-drivers-without-token
    docker compose exec api python scripts/audit_device_tokens.py --deactivate-stale
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from typing import Any

sys.path.insert(0, ".")

from app import create_app
from ext import db
from models import DeviceToken, Driver
from services.notifications.push_token_platform import (
    looks_like_expo_token,
    looks_like_fcm_token,
)

HEALTHY_PUSH_MAX_AGE_DAYS = 7
STALE_INACTIVE_DAYS = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit des tokens push chauffeur")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Rapport JSON + résumé par classe",
    )
    parser.add_argument(
        "--list-drivers-without-token",
        action="store_true",
        help="Chauffeurs actifs sans token HEALTHY",
    )
    parser.add_argument(
        "--deactivate-stale",
        action="store_true",
        help="Désactive les tokens inactifs depuis plus de 90 jours",
    )
    parser.add_argument(
        "--driver-id",
        type=int,
        help="Filtrer sur un chauffeur",
    )
    return parser.parse_args()


def _now() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _token_preview(token: str) -> str:
    return token[:8] if token else ""


def _provider_mismatch(token: DeviceToken) -> bool:
    provider = (token.provider or "expo").lower()
    value = token.token or ""
    if provider == "expo" and looks_like_fcm_token(value):
        return True
    return bool(provider == "fcm" and looks_like_expo_token(value))


def classify_token(token: DeviceToken, *, now: datetime | None = None) -> str:
    now = now or _now()
    if not token.is_active:
        return "INACTIVE"
    if _provider_mismatch(token):
        return "MISMATCH_PROVIDER"
    last_success = _as_utc(token.last_push_success_at)
    last_seen = _as_utc(token.last_seen_at)
    stale_threshold = now - timedelta(days=HEALTHY_PUSH_MAX_AGE_DAYS)
    if last_success and last_success >= stale_threshold:
        return "HEALTHY"
    if last_seen and last_seen >= stale_threshold:
        return "HEALTHY"
    if last_success is None and last_seen is None:
        return "STALE"
    return "STALE"


def serialize_token(token: DeviceToken, classification: str) -> dict[str, Any]:
    return {
        "id": token.id,
        "driver_id": token.driver_id,
        "provider": token.provider,
        "platform": token.platform,
        "is_active": token.is_active,
        "last_seen_at": _as_utc(token.last_seen_at).isoformat()
        if token.last_seen_at
        else None,
        "last_push_success_at": _as_utc(token.last_push_success_at).isoformat()
        if token.last_push_success_at
        else None,
        "last_push_failure_at": _as_utc(token.last_push_failure_at).isoformat()
        if token.last_push_failure_at
        else None,
        "consecutive_push_failures": token.consecutive_push_failures,
        "token_preview": _token_preview(token.token),
        "classification": classification,
    }


def build_report(*, driver_id: int | None = None) -> dict[str, Any]:
    query = DeviceToken.query.filter(DeviceToken.driver_id.isnot(None))
    if driver_id is not None:
        query = query.filter(DeviceToken.driver_id == driver_id)
    tokens = query.order_by(DeviceToken.updated_at.desc()).all()

    by_class: dict[str, list[dict[str, Any]]] = {
        "HEALTHY": [],
        "STALE": [],
        "INACTIVE": [],
        "MISMATCH_PROVIDER": [],
    }
    for token in tokens:
        classification = classify_token(token)
        by_class[classification].append(serialize_token(token, classification))

    summary = {key: len(items) for key, items in by_class.items()}
    return {
        "generated_at": _now().isoformat(),
        "summary": summary,
        "tokens": by_class,
    }


def list_drivers_without_healthy_token() -> list[dict[str, Any]]:
    drivers = Driver.query.filter(Driver.is_active.is_(True)).all()
    missing: list[dict[str, Any]] = []
    for driver in drivers:
        tokens = DeviceToken.query.filter_by(driver_id=driver.id, is_active=True).all()
        has_healthy = any(classify_token(token) == "HEALTHY" for token in tokens)
        if not has_healthy:
            missing.append(
                {
                    "driver_id": driver.id,
                    "active_tokens": len(tokens),
                    "classifications": [classify_token(t) for t in tokens],
                }
            )
    return missing


def deactivate_stale_tokens() -> int:
    now = _now()
    cutoff = now - timedelta(days=STALE_INACTIVE_DAYS)
    tokens = DeviceToken.query.filter(
        DeviceToken.is_active.is_(True),
        DeviceToken.updated_at < cutoff,
    ).all()
    count = 0
    for token in tokens:
        token.is_active = False
        count += 1
    if count:
        db.session.commit()
    return count


def main() -> int:
    args = parse_args()
    if not (args.report or args.list_drivers_without_token or args.deactivate_stale):
        print("Spécifiez --report, --list-drivers-without-token ou --deactivate-stale")
        return 1

    app = create_app()
    with app.app_context():
        if args.report:
            report = build_report(driver_id=args.driver_id)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            print("\nRésumé:")
            for key, value in report["summary"].items():
                print(f"  {key}: {value}")

        if args.list_drivers_without_token:
            missing = list_drivers_without_healthy_token()
            print(json.dumps(missing, indent=2, ensure_ascii=False))
            print(f"\nChauffeurs actifs sans token HEALTHY: {len(missing)}")

        if args.deactivate_stale:
            count = deactivate_stale_tokens()
            print(f"Tokens désactivés (> {STALE_INACTIVE_DAYS}j): {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
