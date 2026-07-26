"""Métriques Prometheus — device health heartbeat chauffeur."""

from __future__ import annotations

import os

_METRICS_ENABLED = os.getenv(
    "DRIVER_DEVICE_HEALTH_METRICS_ENABLED", "true"
).lower() not in (
    "0",
    "false",
    "no",
    "off",
)

_REPORTS = None
_FIX_AGE = None
_STALE = None
_WAKE = None
_REDIS_WRITE = None
_LEGACY_READ = None
_DUAL_WRITE_MISMATCH = None
_FCM_NO_CALLBACK = None
_IOS_HEALTH = None


_APP_VERSION_MIN_PARTS = 2
_STALE_FIX_THRESHOLD_SECONDS = 300


def _norm_app_version(value: str | None) -> str:
    """Major.minor uniquement (faible cardinalité) — ex. '1.42.3' -> '1.42'."""
    if not value:
        return "unknown"
    parts = str(value).strip().split(".")
    if (
        len(parts) >= _APP_VERSION_MIN_PARTS
        and parts[0].isdigit()
        and parts[1].isdigit()
    ):
        return f"{parts[0]}.{parts[1]}"
    return (str(value).strip() or "unknown")[:12]


def _norm_os_major(value: str | None) -> str:
    """Major uniquement (faible cardinalité) — ex. '17.4' -> '17'."""
    if not value:
        return "unknown"
    head = str(value).strip().split(".")[0]
    return head[:8] if head else "unknown"


def _norm_enum(value: str | None, allowed: frozenset[str]) -> str:
    v = (value or "unknown").strip().lower()
    return v if v in allowed else "unknown"


_IOS_ACCURACY_VALUES = frozenset({"full", "reduced", "unknown"})
_IOS_BG_REFRESH_VALUES = frozenset({"available", "denied", "restricted", "unknown"})

try:
    from prometheus_client import Counter, Histogram
except ImportError:
    Counter = None
    Histogram = None

if Counter is not None and _METRICS_ENABLED:
    _REPORTS = Counter(
        "driver_device_health_reports_total",
        "Heartbeats device-health reçus",
        [
            "platform",
            "manufacturer",
            "battery_optimized",
            "constraint_reason",
            "tracking_active",
            "app_version",
            "os_version",
        ],
    )
    _STALE = Counter(
        "driver_device_stale_fix_total",
        "Heartbeats avec last_fix_age > 300s",
        ["manufacturer", "platform", "app_version", "os_version"],
    )
    _IOS_HEALTH = Counter(
        "driver_device_ios_health_total",
        "Signaux background iOS par heartbeat (diagnostic stale)",
        [
            "accuracy_authorization",
            "low_power_mode",
            "background_refresh",
            "native_task_running",
        ],
    )
    _WAKE = Counter(
        "silent_push_wake_total",
        "Pipeline réveil silent push (sent/throttled/failed/acked)",
        ["sync_type", "result"],
    )
    _REDIS_WRITE = Counter(
        "device_health_redis_write_total",
        "Écritures Redis device-health (dual-write migration)",
        ["key"],
    )
    _LEGACY_READ = Counter(
        "device_health_legacy_read_total",
        "Lectures Redis legacy driver:{id}:device_health",
        ["caller"],
    )
    _DUAL_WRITE_MISMATCH = Counter(
        "device_health_dual_write_mismatch_total",
        "Divergence entre clés Redis new vs legacy",
        ["field"],
    )
    _FCM_NO_CALLBACK = Counter(
        "push_fcm_background_handler_no_callback_total",
        "Silent push FCM reçus sans callback métier branché",
        ["platform"],
    )

if Histogram is not None and _METRICS_ENABLED:
    _FIX_AGE = Histogram(
        "driver_device_last_fix_age_seconds",
        "Âge du dernier fix GPS rapporté par le mobile",
        ["manufacturer", "platform", "app_version", "os_version"],
        buckets=(30, 60, 90, 120, 300, 600, 1800, 3600),
    )


def record_device_health_report(
    *,
    manufacturer: str,
    platform: str,
    battery_optimized: bool | None,
    constraint_reason: str | None,
    last_fix_age_seconds: int | None,
    tracking_active: bool | None = None,
    app_version: str | None = None,
    os_version: str | None = None,
    native_task_running: bool | None = None,
    ios_accuracy_authorization: str | None = None,
    ios_low_power_mode: bool | None = None,
    ios_background_refresh_status: str | None = None,
) -> None:
    manuf = (manufacturer or "unknown")[:32]
    plat = (platform or "unknown")[:16]
    appv = _norm_app_version(app_version)
    osv = _norm_os_major(os_version)
    if _REPORTS is not None:
        _REPORTS.labels(
            platform=plat,
            manufacturer=manuf,
            battery_optimized=str(bool(battery_optimized)).lower(),
            constraint_reason=(constraint_reason or "")[:32],
            tracking_active=str(bool(tracking_active)).lower(),
            app_version=appv,
            os_version=osv,
        ).inc()
    if _FIX_AGE is not None and last_fix_age_seconds is not None:
        _FIX_AGE.labels(
            manufacturer=manuf,
            platform=plat,
            app_version=appv,
            os_version=osv,
        ).observe(float(last_fix_age_seconds))
    if (
        _STALE is not None
        and last_fix_age_seconds is not None
        and last_fix_age_seconds > _STALE_FIX_THRESHOLD_SECONDS
    ):
        _STALE.labels(
            manufacturer=manuf, platform=plat, app_version=appv, os_version=osv
        ).inc()
    # Signaux background iOS uniquement (évite du bruit Android, garde la cardinalité basse).
    if _IOS_HEALTH is not None and plat == "ios":
        _IOS_HEALTH.labels(
            accuracy_authorization=_norm_enum(
                ios_accuracy_authorization, _IOS_ACCURACY_VALUES
            ),
            low_power_mode=str(bool(ios_low_power_mode)).lower()
            if ios_low_power_mode is not None
            else "unknown",
            background_refresh=_norm_enum(
                ios_background_refresh_status, _IOS_BG_REFRESH_VALUES
            ),
            native_task_running=str(bool(native_task_running)).lower()
            if native_task_running is not None
            else "unknown",
        ).inc()


def record_silent_push_wake(*, sync_type: str, result: str) -> None:
    if _WAKE is not None:
        _WAKE.labels(
            sync_type=(sync_type or "unknown")[:32],
            result=(result or "unknown")[:32],
        ).inc()


def record_silent_push_wake_legacy_outcome(*, sync_type: str, outcome: str) -> None:
    """Compat release N-1 : mappe outcome vers result."""
    mapped = outcome
    if outcome in ("resync_success", "received"):
        mapped = "acked"
    record_silent_push_wake(sync_type=sync_type, result=mapped)


def record_device_health_redis_write(*, key: str) -> None:
    if _REDIS_WRITE is not None:
        _REDIS_WRITE.labels(key=(key or "unknown")[:16]).inc()


def record_device_health_legacy_read(*, caller: str) -> None:
    if _LEGACY_READ is not None:
        _LEGACY_READ.labels(caller=(caller or "unknown")[:32]).inc()


def record_device_health_dual_write_mismatch(*, field: str) -> None:
    if _DUAL_WRITE_MISMATCH is not None:
        _DUAL_WRITE_MISMATCH.labels(field=(field or "unknown")[:32]).inc()


def record_fcm_background_handler_no_callback(*, platform: str) -> None:
    if _FCM_NO_CALLBACK is not None:
        _FCM_NO_CALLBACK.labels(platform=(platform or "unknown")[:16]).inc()
