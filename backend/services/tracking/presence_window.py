"""Fenêtre de présence GPS entreprise — Europe/Zurich (P0-F TIME).

Bornes figées 07–19, alignées sur le mobile. Ne pas diverger via env.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from zoneinfo import ZoneInfo

BUSINESS_TIME_ZONE = "Europe/Zurich"
PRESENCE_WINDOW_START_HOUR = 7
PRESENCE_WINDOW_END_HOUR = 19

ServiceWindowStatus = Literal["in_window", "mission_override", "off_duty"]

_ZURICH = ZoneInfo(BUSINESS_TIME_ZONE)


def is_within_presence_window(now_utc: datetime | None = None) -> bool:
    """True si l'instant est dans ``[07:00 ; 19:00[`` Europe/Zurich."""
    now = now_utc or datetime.now(UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    local = now.astimezone(_ZURICH)
    return PRESENCE_WINDOW_START_HOUR <= local.hour < PRESENCE_WINDOW_END_HOUR


def resolve_service_window_status(
    *,
    in_window: bool,
    has_active_mission: bool,
) -> ServiceWindowStatus:
    """Statut temporel séparé du statut métier chauffeur.

    - ``in_window`` : horloge dans 07–19
    - ``mission_override`` : hors fenêtre mais mission active (contrat backend)
    - ``off_duty`` : hors fenêtre sans mission
    """
    if in_window:
        return "in_window"
    if has_active_mission:
        return "mission_override"
    return "off_duty"
