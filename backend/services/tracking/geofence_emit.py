"""Émission Socket.IO des événements geofence chauffeur (pickup / dropoff)."""

from __future__ import annotations

from typing import Iterable


def emit_driver_geofence_events(
    *,
    driver_id: int,
    company_id: int | None,
    geofence_events: Iterable[str],
) -> None:
    """Émet ``driver:arrived_at_pickup`` / ``driver:arrived_at_dropoff`` vers la room entreprise."""
    if company_id is None or int(company_id) <= 0:
        return
    try:
        from ext import socketio
    except Exception:
        return
    room = f"company_{int(company_id)}"
    for event in geofence_events:
        if event == "arrived_at_pickup":
            socketio.emit(
                "driver:arrived_at_pickup",
                {"driver_id": driver_id},
                to=room,
            )
        elif event == "arrived_at_dropoff":
            socketio.emit(
                "driver:arrived_at_dropoff",
                {"driver_id": driver_id},
                to=room,
            )
