#!/usr/bin/env python
"""Vérification couverture tokens FCM natifs (Android chauffeur).

Complète `audit_device_tokens.py` en ciblant le cas « Android sans FCM »
(expo_fallback_unreliable) — souvent lié à SHA-1 Play App Signing absent dans Firebase.

Usage (via Docker) :
    docker compose exec api python scripts/verify_fcm_token_coverage.py --report
    docker compose exec api python scripts/verify_fcm_token_coverage.py --driver-id 7514
    docker compose exec api python scripts/verify_fcm_token_coverage.py --android-expo-only
    docker compose exec api python scripts/verify_fcm_token_coverage.py --driver-id 7514 --expect-fcm
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any

sys.path.insert(0, ".")

from app import create_app
from models import DeviceToken, Driver
from scripts.audit_device_tokens import classify_token, serialize_token
from services.notifications.push_device_selection import android_has_expo_only
from services.notifications.push_token_platform import (
    is_android_fcm_registration_token,
    looks_like_fcm_token,
)

FCM_COVERAGE_FCM_NATIVE_OK = "fcm_native_ok"
FCM_COVERAGE_ANDROID_EXPO_ONLY = "android_expo_only"
FCM_COVERAGE_NO_ANDROID_TOKEN = "no_android_token"
FCM_COVERAGE_NO_ACTIVE_TOKEN = "no_active_token"
FCM_COVERAGE_IOS_ONLY = "ios_only"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vérification couverture FCM natif (Android chauffeur)"
    )
    parser.add_argument("--report", action="store_true", help="Rapport agrégé flotte")
    parser.add_argument(
        "--android-expo-only",
        action="store_true",
        help="Liste les chauffeurs Android actifs avec Expo seul (pas de FCM natif)",
    )
    parser.add_argument("--driver-id", type=int, help="Rapport détaillé pour un chauffeur")
    parser.add_argument(
        "--expect-fcm",
        action="store_true",
        help="Code sortie 1 si aucun token FCM Android actif pour --driver-id",
    )
    parser.add_argument(
        "--gate-json",
        action="store_true",
        help="Sortie gate compacte JSON (driver_id, fcm_present, active_provider, status)",
    )
    parser.add_argument(
        "--operational-only",
        action="store_true",
        default=False,
        help="Limiter aux chauffeurs is_active && is_available",
    )
    return parser.parse_args()


def _token_dict(token: DeviceToken) -> dict[str, Any]:
    classification = classify_token(token)
    value = token.token or ""
    return {
        **serialize_token(token, classification),
        "looks_like_fcm": looks_like_fcm_token(value),
        "looks_like_android_fcm_registration": is_android_fcm_registration_token(value),
    }


def resolve_fcm_coverage(active_tokens: list[DeviceToken]) -> str:
    if not active_tokens:
        return FCM_COVERAGE_NO_ACTIVE_TOKEN

    android_tokens = [
        token
        for token in active_tokens
        if (token.platform or "").strip().lower() == "android"
    ]
    if not android_tokens:
        return FCM_COVERAGE_IOS_ONLY

    fcm_tokens = [
        token
        for token in android_tokens
        if (token.provider or "expo").strip().lower() == "fcm"
    ]
    if fcm_tokens:
        return FCM_COVERAGE_FCM_NATIVE_OK

    if android_has_expo_only(active_tokens):
        return FCM_COVERAGE_ANDROID_EXPO_ONLY

    return FCM_COVERAGE_NO_ANDROID_TOKEN


def _recommendations(coverage: str) -> list[str]:
    if coverage == FCM_COVERAGE_FCM_NATIVE_OK:
        return [
            "Token FCM Android actif détecté.",
            "Si push app tuée échoue encore : tester POST /driver/me/test-push puis logcat `driver.push.fcm.*`.",
        ]
    if coverage == FCM_COVERAGE_ANDROID_EXPO_ONLY:
        return [
            "Android n'a qu'un token Expo — livraison app tuée non fiable avec @react-native-firebase/messaging.",
            "Vérifier SHA-1 Play App Signing dans Firebase Console (voir docs/ops/firebase-fcm-sha1-procedure.md).",
            "Sur l'appareil : session chauffeur + disclosure + permission notifications, puis logcat `driver.push.fcm.get_token_start`.",
            "Attendu mobile : `driver.push.fcm.token` (token_present=true) puis POST save-push-token provider=fcm.",
        ]
    if coverage == FCM_COVERAGE_NO_ACTIVE_TOKEN:
        return [
            "Aucun token actif — ouvrir l'app chauffeur, accepter disclosure et notifications.",
        ]
    if coverage == FCM_COVERAGE_IOS_ONLY:
        return [
            "Tokens actifs iOS uniquement — pas de couverture FCM Android pour cet appareil.",
        ]
    return [
        "Tokens Android actifs mais sans FCM ni Expo classifiable — investiguer provider/platform en base.",
    ]


def build_driver_report(driver: Driver) -> dict[str, Any]:
    active_tokens = (
        DeviceToken.query.filter_by(driver_id=driver.id, is_active=True)
        .order_by(DeviceToken.updated_at.desc())
        .all()
    )
    all_tokens = (
        DeviceToken.query.filter_by(driver_id=driver.id)
        .order_by(DeviceToken.updated_at.desc())
        .limit(20)
        .all()
    )
    coverage = resolve_fcm_coverage(active_tokens)
    fcm_android = [
        token
        for token in active_tokens
        if (token.platform or "").lower() == "android"
        and (token.provider or "").lower() == "fcm"
    ]
    return {
        "driver_id": driver.id,
        "is_active": bool(driver.is_active),
        "is_available": bool(driver.is_available),
        "fcm_coverage": coverage,
        "active_tokens_count": len(active_tokens),
        "active_fcm_android_count": len(fcm_android),
        "active_tokens": [_token_dict(token) for token in active_tokens],
        "recent_tokens": [_token_dict(token) for token in all_tokens],
        "recommendations": _recommendations(coverage),
        "checked_at": datetime.now(UTC).isoformat(),
    }


def build_fleet_report(*, operational_only: bool) -> dict[str, Any]:
    query = Driver.query
    if operational_only:
        query = query.filter(Driver.is_active.is_(True), Driver.is_available.is_(True))
    drivers = query.order_by(Driver.id.asc()).all()

    by_coverage: dict[str, list[int]] = {
        FCM_COVERAGE_FCM_NATIVE_OK: [],
        FCM_COVERAGE_ANDROID_EXPO_ONLY: [],
        FCM_COVERAGE_NO_ACTIVE_TOKEN: [],
        FCM_COVERAGE_IOS_ONLY: [],
        FCM_COVERAGE_NO_ANDROID_TOKEN: [],
    }

    for driver in drivers:
        active_tokens = DeviceToken.query.filter_by(driver_id=driver.id, is_active=True).all()
        coverage = resolve_fcm_coverage(active_tokens)
        by_coverage[coverage].append(int(driver.id))

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "operational_only": operational_only,
        "drivers_total": len(drivers),
        "summary": {key: len(value) for key, value in by_coverage.items()},
        "android_expo_only_driver_ids": by_coverage[FCM_COVERAGE_ANDROID_EXPO_ONLY],
        "no_active_token_driver_ids": by_coverage[FCM_COVERAGE_NO_ACTIVE_TOKEN],
        "fcm_native_ok_driver_ids": by_coverage[FCM_COVERAGE_FCM_NATIVE_OK],
    }


def list_android_expo_only(*, operational_only: bool) -> list[dict[str, Any]]:
    report = build_fleet_report(operational_only=operational_only)
    rows: list[dict[str, Any]] = []
    for driver_id in report["android_expo_only_driver_ids"]:
        driver = Driver.query.get(driver_id)
        if driver is None:
            continue
        rows.append(build_driver_report(driver))
    return rows


def build_gate_result(row: dict[str, Any]) -> dict[str, Any]:
    """Résumé compact gate ops (CI / scripts)."""
    driver_id = int(row["driver_id"])
    fcm_present = row["fcm_coverage"] == FCM_COVERAGE_FCM_NATIVE_OK
    active_tokens = row.get("active_tokens") or []
    active_provider = "none"
    if active_tokens:
        active_provider = str(active_tokens[0].get("provider") or "unknown")
    return {
        "driver_id": driver_id,
        "fcm_present": fcm_present,
        "active_provider": active_provider,
        "status": "PASS" if fcm_present else "FAIL",
        "fcm_coverage": row["fcm_coverage"],
        "checked_at": row.get("checked_at"),
    }


def main() -> int:
    args = parse_args()
    if not (args.report or args.android_expo_only or args.driver_id is not None):
        print(
            "Spécifiez --report, --android-expo-only ou --driver-id",
            file=sys.stderr,
        )
        return 1

    app = create_app()
    with app.app_context():
        exit_code = 0

        if args.report:
            report = build_fleet_report(operational_only=args.operational_only)
            print(json.dumps(report, indent=2, ensure_ascii=False))
            print("\nRésumé couverture FCM:")
            for key, value in report["summary"].items():
                print(f"  {key}: {value}")

        if args.android_expo_only:
            rows = list_android_expo_only(operational_only=args.operational_only)
            print(json.dumps(rows, indent=2, ensure_ascii=False))
            print(f"\nChauffeurs Android Expo-only: {len(rows)}")

        if args.driver_id is not None:
            driver = Driver.query.get(args.driver_id)
            if driver is None:
                print(json.dumps({"error": "driver_not_found", "driver_id": args.driver_id}))
                return 1
            row = build_driver_report(driver)
            if args.gate_json:
                gate = build_gate_result(row)
                print(json.dumps(gate, indent=2, ensure_ascii=False))
            else:
                print(json.dumps(row, indent=2, ensure_ascii=False))
            if args.expect_fcm and row["fcm_coverage"] != FCM_COVERAGE_FCM_NATIVE_OK:
                exit_code = 1
                print(
                    f"\nÉCHEC --expect-fcm: fcm_coverage={row['fcm_coverage']}",
                    file=sys.stderr,
                )

        return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
