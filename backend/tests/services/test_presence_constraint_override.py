"""Tests apply_device_health_override (degraded_constrained)."""

from __future__ import annotations

import time

from services.geolocation.presence import (
    apply_device_health_override,
    compute_location_status,
    presence_status_from_location_status,
)


def _fresh_health(**overrides):
    base = {
        "last_heartbeat_at": int(time.time() * 1000),
        "battery_optimized": True,
        "constraint_reason": "samsung_battery_optimized",
        "fgs_running": True,
        "fg_permission": "granted",
        "bg_permission": "denied",
        "gps_provider_enabled": True,
    }
    base.update(overrides)
    return base


def test_online_stays_online_with_fresh_health() -> None:
    presence, location = apply_device_health_override(
        "online",
        "live",
        _fresh_health(),
    )
    assert presence == "online"
    assert location == "live"


def test_offline_fresh_battery_optimized_becomes_degraded_constrained() -> None:
    presence, location = apply_device_health_override(
        "offline",
        "offline",
        _fresh_health(battery_optimized=True),
    )
    assert presence == "degraded_constrained"
    assert location == "degraded_constrained"


def test_offline_no_health_stays_offline() -> None:
    presence, location = apply_device_health_override(
        "offline",
        "offline",
        None,
    )
    assert presence == "offline"
    assert location == "offline"


def test_offline_stale_health_stays_offline() -> None:
    stale_ms = int(time.time() * 1000) - 300_000
    presence, location = apply_device_health_override(
        "offline",
        "offline",
        _fresh_health(last_heartbeat_at=stale_ms),
    )
    assert presence == "offline"
    assert location == "offline"


def test_degraded_fresh_constraint_reason_becomes_degraded_constrained() -> None:
    presence, location = apply_device_health_override(
        "degraded",
        "stale",
        _fresh_health(battery_optimized=False, constraint_reason="doze"),
    )
    assert presence == "degraded_constrained"
    assert location == "degraded_constrained"


def test_offline_fresh_health_no_constraint_stays_offline() -> None:
    presence, location = apply_device_health_override(
        "offline",
        "offline",
        _fresh_health(battery_optimized=False, constraint_reason=None),
    )
    assert presence == "offline"
    assert location == "offline"


def test_compute_location_status_unchanged() -> None:
    """compute_location_status reste pur (pas d'override device)."""
    assert compute_location_status(mode="mission_live", last_seen_seconds=10) == "live"
    assert (
        presence_status_from_location_status(
            compute_location_status(mode="mission_live", last_seen_seconds=500)
        )
        == "offline"
    )
