"""Contrats versionnés partagés pour la surface driver multi-surface."""

from __future__ import annotations

MISSION_STATUS_VERSION = "1.0.0"
MISSION_SNAPSHOT_VERSION = "1.0.0"
DRIVER_SOCKET_CONTRACT_VERSION = "1.0.0"
DRIVER_TRACKING_CONTRACT_VERSION = "1.0.0"

MISSION_STATUS_VALUES = {
    "ASSIGNED",
    "EN_ROUTE",
    "ARRIVED",
    "IN_PROGRESS",
    "COMPLETED",
    "CANCELLED",
    "REASSIGNED",
    "NO_SHOW",
    "FAILED",
}

DRIVER_SOCKET_EVENT_TYPES = {
    "mission_assigned",
    "mission_updated",
    "mission_reassigned",
    "mission_cancelled",
    "mission_status_changed",
    "driver_location_required",
}
