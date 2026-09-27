"""Politique push chauffeur pour booking_updated."""

from __future__ import annotations

from typing import Any

DRIVER_PUSH_FIELDS = frozenset(
    {
        "scheduled_time",
        "pickup_time",
        "appointment_time",
        "arrival_time",
        "return_time",
        "mission_date",
        "scheduled_date",
        "pickup_location",
        "dropoff_location",
        "return_location",
        "return_address",
        "stops",
        "waypoints",
        "route_steps",
        "intermediate_stops",
        "legs",
    }
)

DRIVER_PUSH_STATUS_VALUES = frozenset({"ASSIGNED", "CANCELED"})


def _normalize_status_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        nested = value.get("new") or value.get("to") or value.get("value")
        return _normalize_status_value(nested)
    text = str(value).strip()
    return text.upper() if text else None


def status_change_triggers_driver_push(changes: dict[str, Any] | None) -> bool:
    """True si le changement de statut justifie un push chauffeur (ASSIGNED/CANCELED)."""
    if not changes or "status" not in changes:
        return False
    new_status = _normalize_status_value(changes.get("status"))
    return new_status in DRIVER_PUSH_STATUS_VALUES


def should_send_driver_push_on_booking_updated(
    *,
    notify_driver_push: bool,
    changes: dict[str, Any] | None,
) -> bool:
    """Détermine si un booking_updated doit déclencher un push chauffeur."""
    if not notify_driver_push:
        return False
    changes_keys = set(changes.keys()) if isinstance(changes, dict) else set()
    return bool(changes_keys.intersection(DRIVER_PUSH_FIELDS)) or (
        status_change_triggers_driver_push(changes)
    )
