"""DTO socket booking léger (Sprint 3 phase 2)."""

from __future__ import annotations

import os
from typing import Any

BOOKING_SOCKET_LITE_PAYLOAD = os.environ.get("BOOKING_SOCKET_LITE_PAYLOAD", "0") == "1"

# Champs consommés par le mobile unified-app (driver realtime merge).
_BOOKING_UPDATED_LITE_KEYS = frozenset(
    {
        "id",
        "booking_id",
        "mission_id",
        "status",
        "updated_at",
        "driver_id",
        "company_id",
        "scheduled_time",
        "pickup_time",
        "dropoff_time",
        "event_type",
        "event_id",
        "version",
        "timestamp",
    }
)


def maybe_shrink_booking_socket_payload(
    data: dict[str, Any], event_type: str
) -> dict[str, Any]:
    if not BOOKING_SOCKET_LITE_PAYLOAD:
        return data
    if not event_type.startswith("booking_"):
        return data
    return {k: v for k, v in data.items() if k in _BOOKING_UPDATED_LITE_KEYS}
