"""Référence temporelle unique pour fraîcheur localisation (PR4)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Mapping

from services.geolocation.presence import compute_last_seen_seconds


def resolve_location_freshness_timestamp(loc_data: Mapping[str, Any]) -> str | None:
    """Horodatage ISO de référence pour l'âge affiché : ``recorded_at`` > ``received_at`` > ``ts``."""
    for key in ("recorded_at", "received_at", "ts"):
        raw = loc_data.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if s:
            return s
    return None


def last_seen_seconds_from_location_fields(
    loc_data: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> int | None:
    """Âge en secondes aligné GET Redis : ``resolve_location_freshness_timestamp`` → âge."""
    ref_iso = resolve_location_freshness_timestamp(loc_data)
    if ref_iso is None:
        return None
    return compute_last_seen_seconds(ref_iso, now=now)


def last_seen_seconds_from_db_last_position_update(
    last_position_update: datetime | None,
    *,
    now: datetime,
) -> int | None:
    """Âge secondes depuis ``Driver.last_position_update`` (repli DB sans clé Redis).

    None si pas d'horodatage : le GET reste ``offline`` via ``compute_location_status``.
    """
    if last_position_update is None:
        return None
    lpu = last_position_update
    if lpu.tzinfo is None:
        lpu = lpu.replace(tzinfo=UTC)
    ref = now if now.tzinfo else now.replace(tzinfo=UTC)
    return max(0, int((ref - lpu).total_seconds()))
