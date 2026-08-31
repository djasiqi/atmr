"""Statut de service chauffeur pour annotations flotte.

Historiquement : fenêtre 07–19 Europe/Zurich (P0-F TIME).
Produit SoT (contrat GPS v4) : ``in_window`` reflète ``Driver.is_available``
(en service), plus l'heure. La fenêtre mission ASSIGNED (T−lead / T+grace)
reste dans ``assigned_in_tracking_window`` — ne pas confondre.
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
    """Legacy : True si l'instant est dans ``[07:00 ; 19:00[`` Europe/Zurich.

    Ne plus utiliser comme gate produit pour start/stop GPS.
    Conservé pour telemetry / tests de rétrocompatibilité.
    """
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
    """Statut de service séparé du statut GPS.

    Depuis le contrat GPS v4, ``in_window`` doit être alimenté par
    ``Driver.is_available`` (en service), pas par l'horloge 07–19.

    - ``in_window`` : chauffeur en service
    - ``mission_override`` : hors service mais mission encore active
    - ``off_duty`` : hors service sans mission
    """
    if in_window:
        return "in_window"
    if has_active_mission:
        return "mission_override"
    return "off_duty"
