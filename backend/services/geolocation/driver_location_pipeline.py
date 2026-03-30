"""Pipeline metier unique (P0) - point d'entree partage HTTP + socket.

Les etapes 3-9 (dedup Redis, persistance, fanout, metriques) restent dans les
handlers HTTP / ``chat`` et ``LocationService`` ; la dedup avant store est
``should_skip_location_ingest``. Ici : tri batch monotone (etape 2 du plan).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from services.geolocation.driver_location_dedup import should_skip_location_ingest


def _parse_point_time(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        s = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def sort_location_points_monotone_by_recorded_at(
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Etape 2 - tri monotone par ``recorded_at`` (fallback ``timestamp``)."""

    def _key(p: dict[str, Any]) -> datetime:
        dt = _parse_point_time(p.get("recorded_at")) or _parse_point_time(
            p.get("timestamp")
        )
        return dt if dt is not None else datetime.min.replace(tzinfo=UTC)

    return sorted(points, key=_key)


def process_driver_location_points(
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Etapes 1-2 (ordre plan P0) : copie triee par fraicheur d'enregistrement.

    La dedup par ``location_event_id`` (Redis) puis proximite apres fraicheur
    s'applique point par point via ``should_skip_location_ingest``.
    """
    return sort_location_points_monotone_by_recorded_at(list(points))


__all__ = [
    "process_driver_location_points",
    "should_skip_location_ingest",
    "sort_location_points_monotone_by_recorded_at",
]
