from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Callable

LAT_THRESHOLD = 90.0
LON_THRESHOLD = 180.0


@dataclass(frozen=True, slots=True)
class UpdateDriverLocationCommand:
    driver_id: int
    latitude: float
    longitude: float
    speed: float | None = None
    heading: float | None = None
    accuracy: float | None = None
    ts: str | None = None


@dataclass(frozen=True, slots=True)
class UpdateDriverLocationResult:
    snapped_lat: float
    snapped_lon: float
    source: str
    geofence_events: list[str]


class UpdateDriverLocationUseCase:
    """Use-case Application: mise à jour de localisation chauffeur.

    Ce use-case encapsule la validation d'entrée et délègue l'exécution à une
    dépendance injectée (typiquement `LocationService.update_driver_location`).
    """

    def __init__(
        self,
        *,
        update_location_fn: Callable[..., object],
        now_utc_fn: Callable[[], datetime] | None = None,
    ) -> None:
        super().__init__()
        self._update_location = update_location_fn
        self._now_utc = now_utc_fn or (lambda: datetime.now(UTC))

    def execute(self, cmd: UpdateDriverLocationCommand) -> UpdateDriverLocationResult:
        if not (-LAT_THRESHOLD <= cmd.latitude <= LAT_THRESHOLD) or not (
            -LON_THRESHOLD <= cmd.longitude <= LON_THRESHOLD
        ):
            raise ValueError("Coordinates out of valid range")

        timestamp = self._parse_ts(cmd.ts)

        # On garde la signature la plus permissive possible (typage runtime via attrs).
        res = self._update_location(
            driver_id=cmd.driver_id,
            latitude=cmd.latitude,
            longitude=cmd.longitude,
            speed=cmd.speed,
            heading=cmd.heading,
            accuracy=cmd.accuracy,
            source="gps",
            timestamp=timestamp,
        )

        snapped_lat = getattr(res, "snapped_lat", cmd.latitude)
        snapped_lon = getattr(res, "snapped_lon", cmd.longitude)
        source = getattr(res, "source", "raw")
        geofence_events = list(getattr(res, "geofence_events", []) or [])

        return UpdateDriverLocationResult(
            snapped_lat=float(snapped_lat),
            snapped_lon=float(snapped_lon),
            source=str(source),
            geofence_events=geofence_events,
        )

    def _parse_ts(self, ts: str | None) -> datetime:
        if not ts:
            return self._now_utc()
        try:
            # Support "Z"
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return self._now_utc()
