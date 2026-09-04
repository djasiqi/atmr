"""Ingestion device-health heartbeat + snapshot Redis + métriques."""

from __future__ import annotations

import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from ext import db, redis_client
from models.driver_device_health_event import DriverDeviceHealthEvent

logger = logging.getLogger(__name__)


DEVICE_HEALTH_REDIS_TTL_SEC = int(os.getenv("DEVICE_HEALTH_REDIS_TTL_SEC", "600"))

DEVICE_HEALTH_RETENTION_DAYS = int(os.getenv("DEVICE_HEALTH_RETENTION_DAYS", "30"))


# Champs comparés entre clés new/legacy pendant la migration dual-write.

_DUAL_WRITE_COMPARE_FIELDS = (
    "battery_optimized",
    "constraint_reason",
    "fgs_running",
    "tracking_active",
    "location_permission",
)


def _redis_key_new(driver_id: int) -> str:
    return f"driver:{driver_id}:health"


def _redis_key_legacy(driver_id: int) -> str:
    return f"driver:{int(driver_id)}:device_health"


def _normalize_permission(value: Any) -> str | None:
    if value is None:
        return None

    raw = str(value).strip().lower()

    if raw in {"granted", "always", "when_in_use", "when-in-use"}:
        return "always" if raw in {"always", "granted"} else raw

    if raw in {"denied", "blocked", "restricted"}:
        return "denied"

    if raw in {"undetermined", "not_determined", "prompt"}:
        return "undetermined"

    return raw[:32]


def _resolve_location_permission(payload: dict[str, Any]) -> str | None:
    explicit = payload.get("location_permission")

    if explicit is not None:
        return _normalize_permission(explicit)

    bg = payload.get("bg_permission")

    fg = payload.get("fg_permission")

    if bg == "granted" or bg is True:
        return "always"

    if fg == "granted" or fg is True:
        return "when_in_use"

    if bg == "denied" or fg == "denied":
        return "denied"

    return None


def _bool_to_redis(value: bool | None) -> str:
    if value is None:
        return ""

    return "1" if value else "0"


def _build_legacy_payload(
    *,
    payload: dict[str, Any],
    fgs_running: bool | None,
    battery_optimized: bool | None,
    constraint_reason: str | None,
) -> dict[str, Any]:
    return {
        "fgs_running": fgs_running,
        "battery_optimized": battery_optimized,
        "constraint_reason": constraint_reason,
        "fg_permission": payload.get("fg_permission"),
        "bg_permission": payload.get("bg_permission"),
        "gps_provider_enabled": payload.get("gps_provider_enabled"),
        "battery_level": payload.get("battery_level"),
        "fix_success_rate_last_5min": payload.get("fix_success_rate_last_5min"),
    }


def _compare_dual_write_snapshots(
    *,
    new_snapshot: dict[str, str],
    legacy_mapping: dict[str, str],
) -> None:
    try:
        from services.monitoring.driver_device_health_metrics import (
            record_device_health_dual_write_mismatch,
        )

    except Exception:
        return

    legacy_norm = {
        "battery_optimized": str(legacy_mapping.get("battery_optimized", "")),
        "constraint_reason": str(legacy_mapping.get("constraint_reason", "")),
        "fgs_running": str(legacy_mapping.get("fgs_running", "")),
        "tracking_active": str(legacy_mapping.get("fgs_running", "")),
        "location_permission": "",
    }

    fg = str(legacy_mapping.get("fg_permission", "") or "")

    bg = str(legacy_mapping.get("bg_permission", "") or "")

    if bg in {"granted", "1", "true"}:
        legacy_norm["location_permission"] = "always"

    elif fg in {"granted", "1", "true"}:
        legacy_norm["location_permission"] = "when_in_use"

    for field in _DUAL_WRITE_COMPARE_FIELDS:
        new_val = str(new_snapshot.get(field, "") or "")

        legacy_val = str(legacy_norm.get(field, "") or "")

        if new_val != legacy_val:
            record_device_health_dual_write_mismatch(field=field)

            logger.warning(
                "[device_health] dual-write mismatch driver field=%s new=%s legacy=%s",
                field,
                new_val,
                legacy_val,
            )


def _write_redis_snapshots(
    driver_id: int,
    *,
    new_snapshot: dict[str, str],
    legacy_payload: dict[str, Any],
) -> None:
    if not redis_client:
        return

    try:
        from services.monitoring.driver_device_health_metrics import (
            record_device_health_redis_write,
        )

    except Exception:
        record_device_health_redis_write = None  # type: ignore[assignment]

    new_key = _redis_key_new(driver_id)

    try:
        redis_client.hset(new_key, mapping=new_snapshot)

        redis_client.expire(new_key, DEVICE_HEALTH_REDIS_TTL_SEC)

        if record_device_health_redis_write:
            record_device_health_redis_write(key="new")

    except Exception as exc:
        logger.debug(
            "[device_health] redis new snapshot failed driver=%s: %s", driver_id, exc
        )

    try:
        from services.geolocation.device_health import write_device_health

        legacy_mapping = {
            "last_heartbeat_at": str(int(time.time() * 1000)),
            "fgs_running": _bool_to_redis(legacy_payload.get("fgs_running")),
            "battery_optimized": _bool_to_redis(
                legacy_payload.get("battery_optimized")
            ),
            "constraint_reason": str(legacy_payload.get("constraint_reason") or ""),
            "fg_permission": str(legacy_payload.get("fg_permission") or ""),
            "bg_permission": str(legacy_payload.get("bg_permission") or ""),
            "gps_provider_enabled": _bool_to_redis(
                legacy_payload.get("gps_provider_enabled")
            ),
            "battery_level": str(legacy_payload.get("battery_level") or ""),
            "fix_success_rate_last_5min": str(
                legacy_payload.get("fix_success_rate_last_5min") or ""
            ),
        }

        write_device_health(
            redis_client,
            driver_id,
            legacy_payload,
            ttl_sec=min(DEVICE_HEALTH_REDIS_TTL_SEC, 120),
        )

        if record_device_health_redis_write:
            record_device_health_redis_write(key="legacy")

        _compare_dual_write_snapshots(
            new_snapshot=new_snapshot, legacy_mapping=legacy_mapping
        )

    except Exception as exc:
        logger.debug(
            "[device_health] redis legacy snapshot failed driver=%s: %s", driver_id, exc
        )


def _parse_tracking_pipeline(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Extrait et borne `tracking_pipeline` (backward-compatible, optionnel)."""
    raw = payload.get("tracking_pipeline")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    # Borne défensive : éviter payloads aberrants en prod.
    if len(raw) > 64:
        raw = dict(list(raw.items())[:64])
    return raw


def ingest_driver_device_health(
    driver_id: int,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Persiste un event device-health + snapshot Redis (dual-write migration)."""

    now = datetime.now(UTC)

    manufacturer = str(payload.get("manufacturer") or "").strip() or None

    model = str(payload.get("model") or "").strip() or None

    platform = str(payload.get("platform") or payload.get("os") or "").strip() or None

    battery_optimized = payload.get("battery_optimized")

    if battery_optimized is not None:
        battery_optimized = bool(battery_optimized)

    notifications_enabled = payload.get("notifications_enabled")

    if notifications_enabled is not None:
        notifications_enabled = bool(notifications_enabled)

    tracking_active = payload.get("tracking_active")

    if tracking_active is None:
        tracking_active = payload.get("fgs_running")

    if tracking_active is not None:
        tracking_active = bool(tracking_active)

    app_state = str(payload.get("app_state") or "").strip() or None

    def _optional_int(key: str) -> int | None:
        raw = payload.get(key)
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None

    # GNSS : préférer location_fix_age_seconds (autorité Location.timestamp)
    location_fix_age_seconds = _optional_int("location_fix_age_seconds")
    last_fix_age_seconds = (
        location_fix_age_seconds
        if location_fix_age_seconds is not None
        else _optional_int("last_fix_age_seconds")
    )

    constraint_reason = str(payload.get("constraint_reason") or "").strip() or None

    fgs_running = payload.get("fgs_running")

    if fgs_running is not None:
        fgs_running = bool(fgs_running)

    trigger_reason = str(payload.get("trigger_reason") or "").strip() or None

    location_permission = _resolve_location_permission(payload)

    native_start_phase = str(payload.get("native_start_phase") or "").strip() or None
    native_start_error = str(payload.get("native_start_error") or "").strip() or None
    if native_start_error:
        native_start_error = native_start_error[:512]

    def _optional_bool(key: str) -> bool | None:
        raw = payload.get(key)
        if raw is None:
            return None
        return bool(raw)

    native_task_defined = _optional_bool("native_task_defined")
    native_started_before = _optional_bool("native_started_before")
    native_started_after = _optional_bool("native_started_after")

    # --- Diagnostic Lot 1 (observabilité device-health enrichie) ---
    app_version = str(payload.get("app_version") or "").strip() or None
    if app_version:
        app_version = app_version[:32]
    os_version = str(payload.get("os_version") or "").strip() or None
    if os_version:
        os_version = os_version[:32]

    def _optional_str(key: str, max_len: int) -> str | None:
        raw = str(payload.get(key) or "").strip() or None
        return raw[:max_len] if raw else None

    native_build_version = _optional_str("native_build_version", 32)
    expo_runtime_version = _optional_str("expo_runtime_version", 32)
    ota_update_id = _optional_str("ota_update_id", 128)
    release_channel = _optional_str("release_channel", 64)
    release_sha = _optional_str("release_sha", 64)

    # Task invoke (≠ GNSS) : préférer task_invoke_age ; compat native_last_fix
    task_invoke_age_seconds = _optional_int("task_invoke_age_seconds")
    native_last_fix_age_seconds = (
        task_invoke_age_seconds
        if task_invoke_age_seconds is not None
        else _optional_int("native_last_fix_age_seconds")
    )
    if task_invoke_age_seconds is None:
        task_invoke_age_seconds = native_last_fix_age_seconds

    observability_class = str(payload.get("observability_class") or "").strip() or None
    if observability_class:
        observability_class = observability_class[:32]

    watch_callback_age_seconds = _optional_int("watch_callback_age_seconds")
    oldest_queue_item_age_seconds = _optional_int("oldest_queue_item_age_seconds")
    persistence_lag_seconds = _optional_int("persistence_lag_seconds")

    native_task_running = _optional_bool("native_task_running")

    ios_accuracy_authorization = (
        str(payload.get("ios_accuracy_authorization") or "").strip() or None
    )
    if ios_accuracy_authorization:
        ios_accuracy_authorization = ios_accuracy_authorization[:16]

    ios_low_power_mode = _optional_bool("ios_low_power_mode")

    ios_background_refresh_status = (
        str(payload.get("ios_background_refresh_status") or "").strip() or None
    )
    if ios_background_refresh_status:
        ios_background_refresh_status = ios_background_refresh_status[:16]

    tracking_pipeline = _parse_tracking_pipeline(payload)

    event = DriverDeviceHealthEvent(
        driver_id=driver_id,
        recorded_at=now,
        manufacturer=manufacturer,
        model=model,
        platform=platform,
        battery_optimized=battery_optimized,
        location_permission=location_permission,
        notifications_enabled=notifications_enabled,
        tracking_active=tracking_active,
        app_state=app_state,
        last_fix_age_seconds=last_fix_age_seconds,
        constraint_reason=constraint_reason,
        fgs_running=fgs_running,
        trigger_reason=trigger_reason,
        native_start_phase=native_start_phase,
        native_start_error=native_start_error,
        native_task_defined=native_task_defined,
        native_started_before=native_started_before,
        native_started_after=native_started_after,
        app_version=app_version,
        os_version=os_version,
        native_last_fix_age_seconds=native_last_fix_age_seconds,
        native_task_running=native_task_running,
        ios_accuracy_authorization=ios_accuracy_authorization,
        ios_low_power_mode=ios_low_power_mode,
        ios_background_refresh_status=ios_background_refresh_status,
        native_build_version=native_build_version,
        expo_runtime_version=expo_runtime_version,
        ota_update_id=ota_update_id,
        release_channel=release_channel,
        release_sha=release_sha,
        tracking_pipeline=tracking_pipeline,
    )

    db.session.add(event)

    snapshot = {
        "driver_id": str(driver_id),
        "recorded_at": now.isoformat(),
        "manufacturer": manufacturer or "",
        "model": model or "",
        "platform": platform or "",
        "battery_optimized": _bool_to_redis(battery_optimized),
        "location_permission": location_permission or "",
        "notifications_enabled": _bool_to_redis(notifications_enabled),
        "tracking_active": _bool_to_redis(tracking_active),
        "app_state": app_state or "",
        "last_fix_age_seconds": str(
            last_fix_age_seconds if last_fix_age_seconds is not None else ""
        ),
        "location_fix_age_seconds": str(
            location_fix_age_seconds
            if location_fix_age_seconds is not None
            else (last_fix_age_seconds if last_fix_age_seconds is not None else "")
        ),
        "constraint_reason": constraint_reason or "",
        "fgs_running": _bool_to_redis(fgs_running),
        "trigger_reason": trigger_reason or "",
        "native_start_phase": native_start_phase or "",
        "native_start_error": native_start_error or "",
        "native_task_defined": _bool_to_redis(native_task_defined),
        "native_started_before": _bool_to_redis(native_started_before),
        "native_started_after": _bool_to_redis(native_started_after),
        "app_version": app_version or "",
        "os_version": os_version or "",
        "native_build_version": native_build_version or "",
        "expo_runtime_version": expo_runtime_version or "",
        "ota_update_id": ota_update_id or "",
        "release_channel": release_channel or "",
        "release_sha": release_sha or "",
        "task_invoke_age_seconds": str(
            task_invoke_age_seconds if task_invoke_age_seconds is not None else ""
        ),
        # Compat lecture dashboards / alertes (alias task_invoke)
        "native_last_fix_age_seconds": str(
            native_last_fix_age_seconds
            if native_last_fix_age_seconds is not None
            else ""
        ),
        "observability_class": observability_class or "",
        "watch_callback_age_seconds": str(
            watch_callback_age_seconds if watch_callback_age_seconds is not None else ""
        ),
        "oldest_queue_item_age_seconds": str(
            oldest_queue_item_age_seconds
            if oldest_queue_item_age_seconds is not None
            else ""
        ),
        "persistence_lag_seconds": str(
            persistence_lag_seconds if persistence_lag_seconds is not None else ""
        ),
        "native_task_running": _bool_to_redis(native_task_running),
        "ios_accuracy_authorization": ios_accuracy_authorization or "",
        "ios_low_power_mode": _bool_to_redis(ios_low_power_mode),
        "ios_background_refresh_status": ios_background_refresh_status or "",
        "last_heartbeat_at": str(int(now.timestamp() * 1000)),
    }
    if tracking_pipeline is not None:
        snapshot["tracking_pipeline"] = tracking_pipeline
        snapshot["pipeline_snapshot_version"] = str(
            tracking_pipeline.get("pipeline_snapshot_version") or ""
        )

    legacy_payload = _build_legacy_payload(
        payload=payload,
        fgs_running=fgs_running,
        battery_optimized=battery_optimized,
        constraint_reason=constraint_reason,
    )

    _write_redis_snapshots(
        driver_id, new_snapshot=snapshot, legacy_payload=legacy_payload
    )

    try:
        from services.monitoring.driver_device_health_metrics import (
            record_device_health_report,
        )

        record_device_health_report(
            manufacturer=manufacturer or "unknown",
            platform=platform or "unknown",
            battery_optimized=battery_optimized,
            constraint_reason=constraint_reason,
            last_fix_age_seconds=last_fix_age_seconds,
            tracking_active=tracking_active,
            app_version=app_version,
            os_version=os_version,
            native_task_running=native_task_running,
            ios_accuracy_authorization=ios_accuracy_authorization,
            ios_low_power_mode=ios_low_power_mode,
            ios_background_refresh_status=ios_background_refresh_status,
        )

    except Exception:
        pass

    db.session.commit()

    result = dict(snapshot)
    if tracking_pipeline is not None:
        result["tracking_pipeline"] = tracking_pipeline
        event_id = getattr(event, "id", None)
        if event_id is not None:
            result["device_health_event_id"] = event_id
    return result


def _parse_bool_redis(value: Any) -> bool | None:
    if value is None:
        return None
    raw = str(value).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", ""}:
        return False
    return None


def parse_driver_device_health_snapshot(
    raw: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Normalise le hash Redis canonique pour presence override + dashboard."""
    if not raw:
        return None
    last_hb = raw.get("last_heartbeat_at") or raw.get("recorded_at")
    try:
        last_heartbeat_at = int(last_hb) if last_hb not in (None, "") else None
    except (TypeError, ValueError):
        last_heartbeat_at = None
    if last_heartbeat_at is not None and last_heartbeat_at < 10_000_000_000:
        last_heartbeat_at = int(last_heartbeat_at * 1000)
    return {
        "last_heartbeat_at": last_heartbeat_at,
        "fgs_running": _parse_bool_redis(raw.get("fgs_running")),
        "battery_optimized": _parse_bool_redis(raw.get("battery_optimized")),
        "constraint_reason": (str(raw.get("constraint_reason") or "").strip() or None),
        "location_permission": (
            str(raw.get("location_permission") or "").strip() or None
        ),
        "tracking_active": _parse_bool_redis(raw.get("tracking_active")),
        "platform": (str(raw.get("platform") or "").strip() or None),
        "manufacturer": (str(raw.get("manufacturer") or "").strip() or None),
        "last_fix_age_seconds": raw.get("last_fix_age_seconds"),
    }


def read_driver_device_health_snapshot(driver_id: int) -> dict[str, Any] | None:
    if not redis_client:
        return None

    try:
        raw = redis_client.hgetall(_redis_key_new(driver_id))

        if not raw:
            return None

        out: dict[str, Any] = {}

        for k, v in raw.items():
            key = k.decode() if isinstance(k, bytes) else str(k)

            val = v.decode() if isinstance(v, bytes) else str(v)

            out[key] = val

        return parse_driver_device_health_snapshot(out)

    except Exception:
        return None


def read_driver_device_health_batch(
    driver_ids: list[int] | tuple[int, ...],
) -> dict[int, dict[str, Any] | None]:
    """Lit les snapshots device-health canoniques (driver:{id}:health) en pipeline."""

    out: dict[int, dict[str, Any] | None] = {int(d): None for d in driver_ids}

    if not redis_client or not driver_ids:
        return out

    try:
        pipe = redis_client.pipeline()

        for did in driver_ids:
            pipe.hgetall(_redis_key_new(int(did)))

        results = pipe.execute()

    except Exception as exc:
        logger.debug("[device_health] batch read failed: %s", exc)

        return out

    for did, raw in zip(driver_ids, results, strict=True):
        if not raw:
            continue

        parsed: dict[str, Any] = {}

        for k, v in raw.items():
            key = k.decode() if isinstance(k, bytes) else str(k)

            val = v.decode() if isinstance(v, bytes) else str(v)

            parsed[key] = val

        out[int(did)] = parse_driver_device_health_snapshot(parsed)

    return out


def purge_old_device_health_events() -> int:
    """Supprime les events plus vieux que DEVICE_HEALTH_RETENTION_DAYS."""

    cutoff = datetime.now(UTC) - timedelta(days=DEVICE_HEALTH_RETENTION_DAYS)

    deleted = DriverDeviceHealthEvent.query.filter(
        DriverDeviceHealthEvent.recorded_at < cutoff
    ).delete(synchronize_session=False)

    db.session.commit()

    return int(deleted or 0)


def resolve_tracking_display_status(
    *,
    location_status: str,
    health_snapshot: dict[str, Any] | None,
) -> str:
    """4 états carte : live | stale | degraded_constrained | offline_unknown."""

    if location_status in {"live", "recent"}:
        return "live"

    if location_status == "stale":
        return "stale"

    if location_status != "offline":
        return location_status

    if not health_snapshot:
        return "offline_unknown"

    constraint = str(health_snapshot.get("constraint_reason") or "").lower()

    raw_batt = health_snapshot.get("battery_optimized")
    if isinstance(raw_batt, bool):
        battery_opt = raw_batt
    else:
        battery_opt = str(raw_batt or "0") == "1"

    if battery_opt or constraint in {
        "battery_optimized",
        "permission_bg_denied",
        "permission_fg_denied",
        "fgs_not_running",
        "gps_provider_disabled",
    }:
        return "degraded_constrained"

    return "offline_unknown"
