"""Adapter pour LocationService (compatibilité temporaire)."""

from __future__ import annotations

from typing import Any, Callable

from services.geolocation.location import LocationService, get_location_service


class LocationServiceAdapter:
    """Adapter qui encapsule LocationService pour l'utiliser avec
    UpdateDriverLocationUseCase.

    Permet d'utiliser LocationService existant avec les nouveaux use-cases.
    """

    def __init__(  # pyright: ignore[reportMissingSuperCall]
        self, location_service: LocationService | None = None
    ) -> None:
        """Initialise l'adapter.

        Args:
            location_service: Instance de LocationService (créé par défaut si None).
        """
        self.location_service = location_service or get_location_service()

    def update_driver_location(
        self,
        driver_id: int,
        latitude: float,
        longitude: float,
        speed: float | None = None,
        heading: float | None = None,
        accuracy: float | None = None,
        source: str = "gps",
        timestamp: Any | None = None,
        **extra: Any,
    ) -> Any:
        """Met à jour la localisation d'un chauffeur via LocationService.

        Returns:
            LocationUpdateResult avec snapped_lat, snapped_lon, source, geofence_events.
        """
        return self.location_service.update_driver_location(
            driver_id=driver_id,
            latitude=latitude,
            longitude=longitude,
            speed=speed,
            heading=heading,
            accuracy=accuracy,
            source=source,
            timestamp=timestamp,
            **extra,
        )


def create_location_update_fn() -> Callable[..., Any]:
    """Factory function pour créer une fonction de mise à jour de localisation.

    Returns:
        Fonction compatible avec UpdateDriverLocationUseCase.
    """
    adapter = LocationServiceAdapter()
    return adapter.update_driver_location
