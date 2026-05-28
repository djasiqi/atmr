"""Événements contractuels (aligné docs/ops/ws-event-contract.md)."""

from __future__ import annotations

CRITICAL_EVENT_TYPES = frozenset(
    {
        "booking_updated",
        "booking_cancelled",
        "team_chat_message",
        "dispatch_assignment",
        "dispatch_run_started",
        "dispatch_run_completed",
        "dispatch_run_failed",
        "urgent_alert",
        "delay_live_invalidate",
    }
)

HIGH_EVENT_TYPES = frozenset(
    {
        "driver_location_update",
        "driver_live_state_update",
    }
)


def event_criticality(event_type: str) -> str:
    if event_type in CRITICAL_EVENT_TYPES or event_type.startswith("dispatch"):
        return "critical"
    if event_type in HIGH_EVENT_TYPES:
        return "high"
    return "normal"
