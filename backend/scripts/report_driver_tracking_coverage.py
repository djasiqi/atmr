#!/usr/bin/env python3
"""Rapport couverture tracking GPS par chauffeur (P0-B).

Usage::
    docker compose exec api python -m scripts.report_driver_tracking_coverage --days 7
    docker compose exec api python -m scripts.report_driver_tracking_coverage --days 7 --output /tmp/coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

if __name__ == "__main__" and __package__ is None:  # pragma: no cover
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))

from app import create_app
from models import DeviceToken, Driver
from models.driver_device_health_event import DriverDeviceHealthEvent

ROOT_CAUSE_VALUES = (
    "fgs_not_running",
    "bg_permission_denied",
    "no_push_token",
    "app_not_active",
    "work_window_closed",
    "ios_only_gap",
    "version_outdated",
    "battery_optimized",
    "investigation_required",
    "tracking_ok",
)

CSV_FIELDS = [
    "driver_id",
    "company_id",
    "app_version",
    "os",
    "last_gps_at",
    "last_gps_coords",
    "push_token_present",
    "bg_permission",
    "fgs_android_running",
    "positions_24h",
    "positions_7d",
    "in_tracking_pipeline",
    "root_cause",
    "action_plan",
]

STALE_AUDIT_FIELDS = [
    "driver_id",
    "manufacturer",
    "model",
    "platform",
    "app_version",
    "os_version",
    "heartbeats",
    "stale_count",
    "stale_rate_pct",
    "last_recorded_at",
    "last_gps_timestamp",
    "last_heartbeat_timestamp",
    "last_received_timestamp",
    "last_fix_age_seconds",
    "native_last_fix_age_seconds",
]

_STALE_THRESHOLD_SECONDS = 300


def _read_device_health(driver_id: int) -> dict[str, Any]:
    try:
        from ext import redis_client

        if redis_client is None:
            return {}
        raw = redis_client.get(f"driver:{driver_id}:device_health")
        if not raw:
            raw = redis_client.hgetall(f"driver:{driver_id}:device_health")
            if isinstance(raw, dict) and raw:
                decoded: dict[str, Any] = {}
                for k, v in raw.items():
                    key = k.decode() if isinstance(k, bytes) else str(k)
                    val = v.decode() if isinstance(v, bytes) else v
                    decoded[key] = val
                return decoded
            return {}
        if isinstance(raw, bytes):
            raw = raw.decode()
        return json.loads(raw) if isinstance(raw, str) else {}
    except Exception:
        return {}


def _count_stream_positions(driver_id: int, *, since: datetime) -> int:
    try:
        from ext import redis_client

        if redis_client is None:
            return 0
        count = 0
        entries = redis_client.xrevrange(
            "driver_location_stream", count=5000, max="+", min="-"
        )
        for _entry_id, fields in entries:
            if not isinstance(fields, dict):
                continue
            did_raw = fields.get(b"driver_id") or fields.get("driver_id")
            if did_raw is None:
                continue
            did = int(did_raw.decode() if isinstance(did_raw, bytes) else did_raw)
            if did != driver_id:
                continue
            ts_raw = fields.get(b"ts") or fields.get("ts")
            if ts_raw is None:
                count += 1
                continue
            ts_str = ts_raw.decode() if isinstance(ts_raw, bytes) else str(ts_raw)
            try:
                ts_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=UTC)
                if ts_dt >= since:
                    count += 1
            except Exception:
                count += 1
        return count
    except Exception:
        return 0


def _infer_root_cause(
    *,
    in_pipeline: bool,
    push_token_present: bool,
    health: dict[str, Any],
    last_gps_at: datetime | None,
    now: datetime,
) -> tuple[str, str]:
    if in_pipeline:
        return "tracking_ok", "Aucune action — positions reçues sur la fenêtre."

    if not push_token_present:
        return (
            "no_push_token",
            "Ré-enregistrer le token push (onboarding / réouverture app).",
        )

    platform = str(health.get("platform") or health.get("os") or "").lower()
    bg = str(health.get("background_permission") or health.get("bg_permission") or "")
    if bg.lower() in {"denied", "false", "0"}:
        return (
            "bg_permission_denied",
            "Guider le chauffeur vers les permissions localisation « Toujours ».",
        )

    if platform == "android":
        fgs = health.get("foreground_service_running")
        if fgs is None:
            fgs = health.get("fgs_running")
        if fgs is False or str(fgs).lower() == "false":
            return (
                "fgs_not_running",
                "Vérifier FGS Android + bannière DriverTrackingReadinessGate.",
            )

    if str(health.get("battery_optimized", "")).lower() in {"true", "1"}:
        return (
            "battery_optimized",
            "Désactiver optimisation batterie OEM pour Lirie.",
        )

    if last_gps_at is None:
        return (
            "app_not_active",
            "Vérifier que l'app est ouverte en journée ou mission active.",
        )

    age_days = (now - last_gps_at).total_seconds() / 86400.0
    if age_days > 7:
        return (
            "app_not_active",
            "Aucune position 7j — vérifier usage app et permissions.",
        )

    return (
        "investigation_required",
        "Analyser logs HTTP User-Agent et device_health sous 48h.",
    )


def build_coverage_rows(*, days: int) -> list[dict[str, Any]]:
    now = datetime.now(UTC)
    since_7d = now - timedelta(days=days)
    since_24h = now - timedelta(hours=24)
    rows: list[dict[str, Any]] = []

    flask_app = create_app()
    with flask_app.app_context():
        drivers = (
            Driver.query.filter(Driver.is_active.is_(True))
            .order_by(Driver.id.asc())
            .all()
        )
        for driver in drivers:
            driver_id = int(driver.id)
            health = _read_device_health(driver_id)
            last_gps_at = getattr(driver, "last_position_update", None)
            if last_gps_at is not None and last_gps_at.tzinfo is None:
                last_gps_at = last_gps_at.replace(tzinfo=UTC)

            lat = getattr(driver, "latitude", None)
            lon = getattr(driver, "longitude", None)
            coords = (
                f"{float(lat):.5f},{float(lon):.5f}"
                if lat is not None and lon is not None
                else ""
            )

            push_token_present = (
                DeviceToken.query.filter_by(driver_id=driver_id, is_active=True).count()
                > 0
            )

            pos_24h = _count_stream_positions(driver_id, since=since_24h)
            pos_7d = _count_stream_positions(driver_id, since=since_7d)
            if pos_7d == 0 and last_gps_at and last_gps_at >= since_7d:
                pos_7d = 1
            if pos_24h == 0 and last_gps_at and last_gps_at >= since_24h:
                pos_24h = 1

            in_pipeline = pos_7d > 0 or (
                last_gps_at is not None and last_gps_at >= since_7d
            )

            root_cause, action_plan = _infer_root_cause(
                in_pipeline=in_pipeline,
                push_token_present=push_token_present,
                health=health,
                last_gps_at=last_gps_at,
                now=now,
            )

            rows.append(
                {
                    "driver_id": driver_id,
                    "company_id": getattr(driver, "company_id", None),
                    "app_version": health.get("app_version") or "",
                    "os": health.get("platform") or health.get("os") or "",
                    "last_gps_at": (last_gps_at.isoformat() if last_gps_at else ""),
                    "last_gps_coords": coords,
                    "push_token_present": "oui" if push_token_present else "non",
                    "bg_permission": health.get("background_permission")
                    or health.get("bg_permission")
                    or "",
                    "fgs_android_running": health.get("foreground_service_running", ""),
                    "positions_24h": pos_24h,
                    "positions_7d": pos_7d,
                    "in_tracking_pipeline": "oui" if in_pipeline else "non",
                    "root_cause": root_cause,
                    "action_plan": action_plan,
                }
            )

    return rows


def build_stale_audit_rows(*, hours: int) -> list[dict[str, Any]]:
    """Audit A0 : taux stale par chauffeur depuis driver_device_health_events."""
    now = datetime.now(UTC)
    since = now - timedelta(hours=hours)
    rows: list[dict[str, Any]] = []

    flask_app = create_app()
    with flask_app.app_context():
        events = (
            DriverDeviceHealthEvent.query.filter(
                DriverDeviceHealthEvent.recorded_at >= since
            )
            .order_by(
                DriverDeviceHealthEvent.driver_id.asc(),
                DriverDeviceHealthEvent.recorded_at.desc(),
            )
            .all()
        )
        by_driver: dict[int, list[DriverDeviceHealthEvent]] = {}
        for ev in events:
            by_driver.setdefault(int(ev.driver_id), []).append(ev)

        for driver_id, evs in sorted(by_driver.items()):
            stale_count = sum(
                1
                for ev in evs
                if ev.last_fix_age_seconds is not None
                and ev.last_fix_age_seconds > _STALE_THRESHOLD_SECONDS
            )
            heartbeats = len(evs)
            latest = evs[0]
            recorded_at = latest.recorded_at
            if recorded_at is not None and recorded_at.tzinfo is None:
                recorded_at = recorded_at.replace(tzinfo=UTC)
            fix_age = latest.last_fix_age_seconds or 0
            gps_ts = (
                recorded_at - timedelta(seconds=fix_age)
                if recorded_at is not None
                else None
            )
            rows.append(
                {
                    "driver_id": driver_id,
                    "manufacturer": latest.manufacturer or "",
                    "model": latest.model or "",
                    "platform": latest.platform or "",
                    "app_version": latest.app_version or "",
                    "os_version": latest.os_version or "",
                    "heartbeats": heartbeats,
                    "stale_count": stale_count,
                    "stale_rate_pct": round(
                        100.0 * stale_count / heartbeats if heartbeats else 0.0, 1
                    ),
                    "last_recorded_at": (
                        recorded_at.isoformat() if recorded_at else ""
                    ),
                    "last_gps_timestamp": (gps_ts.isoformat() if gps_ts else ""),
                    "last_heartbeat_timestamp": (
                        recorded_at.isoformat() if recorded_at else ""
                    ),
                    "last_received_timestamp": (
                        recorded_at.isoformat() if recorded_at else ""
                    ),
                    "last_fix_age_seconds": latest.last_fix_age_seconds or "",
                    "native_last_fix_age_seconds": (
                        latest.native_last_fix_age_seconds or ""
                    ),
                }
            )

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rapport couverture tracking chauffeurs"
    )
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--stale-audit",
        action="store_true",
        help="Audit A0 : taux stale par chauffeur (driver_device_health_events)",
    )
    parser.add_argument(
        "--stale-hours",
        type=int,
        default=24,
        help="Fenêtre en heures pour --stale-audit (défaut 24)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Fichier CSV (sinon stdout)",
    )
    args = parser.parse_args()

    if args.stale_audit:
        rows = build_stale_audit_rows(hours=args.stale_hours)
        fields = STALE_AUDIT_FIELDS
    else:
        rows = build_coverage_rows(days=args.days)
        fields = CSV_FIELDS

    if args.output:
        with Path(args.output).open("w", newline="", encoding="utf-8") as out:
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    if args.stale_audit:
        high_stale = [r for r in rows if float(r["stale_rate_pct"]) >= 50.0]
        print(
            f"# stale-audit drivers={len(rows)} high_stale_50pct={len(high_stale)} window_h={args.stale_hours}",
            file=sys.stderr,
        )
        return 0

    absent = [r for r in rows if r["in_tracking_pipeline"] == "non"]
    unknown = [r for r in absent if r["root_cause"] in {"", "investigation_required"}]
    print(
        f"# summary drivers={len(rows)} absent={len(absent)} unknown_cause={len(unknown)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
