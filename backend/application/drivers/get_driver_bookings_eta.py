from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Sequence


@dataclass(frozen=True, slots=True)
class BookingEtaItem:
    id: int
    eta_to_pickup_seconds: int | None
    eta_to_dropoff_seconds: (
        int | None
    )  # Après pickup : temps restant jusqu'à destination
    duration_seconds: int | None
    distance_meters: float | int | None
    estimated_arrival: str | None  # ETA au point de prise en charge (avant pickup)
    estimated_arrival_dropoff: (
        str | None
    )  # ETA à destination (après pickup, client à bord)


@dataclass(frozen=True, slots=True)
class DriverBookingsEtaResponse:
    has_gps: bool
    driver_position: dict[str, float] | None
    bookings: list[BookingEtaItem]


class GetDriverBookingsETAUseCase:
    """Use-case Application: calcul ETA pour les bookings du chauffeur."""

    def __init__(
        self,
        *,
        eta_seconds_fn: Callable[[tuple[float, float], tuple[float, float]], int],
        now_local_fn: Callable[[], datetime],
    ) -> None:
        super().__init__()
        self._eta_seconds = eta_seconds_fn
        self._now_local = now_local_fn

    def execute(
        self,
        *,
        driver_lat: float | None,
        driver_lon: float | None,
        bookings: Sequence[Any],
    ) -> DriverBookingsEtaResponse:
        if driver_lat is None or driver_lon is None:
            return DriverBookingsEtaResponse(
                has_gps=False,
                driver_position=None,
                bookings=[
                    BookingEtaItem(
                        id=int(b.id),
                        eta_to_pickup_seconds=None,
                        eta_to_dropoff_seconds=None,
                        duration_seconds=b.duration_seconds,
                        distance_meters=b.distance_meters,
                        estimated_arrival=None,
                        estimated_arrival_dropoff=None,
                    )
                    for b in bookings
                ],
            )

        driver_pos = (float(driver_lat), float(driver_lon))
        current_time = self._now_local()

        def _is_in_progress(s: Any) -> bool:
            raw = getattr(s, "value", s)
            normalized = str(raw).upper() if raw is not None else ""
            return normalized == "IN_PROGRESS"

        items: list[BookingEtaItem] = []
        for b in bookings:
            booking_id = int(b.id)
            pickup_lat = b.pickup_lat
            pickup_lon = b.pickup_lon
            dropoff_lat = b.dropoff_lat
            dropoff_lon = b.dropoff_lon
            status = b.status
            in_progress = _is_in_progress(status)

            eta_to_pickup: int | None = None
            eta_to_dropoff: int | None = None
            total_duration = b.duration_seconds

            if pickup_lat is not None and pickup_lon is not None:
                try:
                    pickup_pos = (float(pickup_lat), float(pickup_lon))
                    eta_to_pickup = int(self._eta_seconds(driver_pos, pickup_pos))

                    if (
                        dropoff_lat is not None
                        and dropoff_lon is not None
                        and not in_progress
                    ):
                        dropoff_pos = (float(dropoff_lat), float(dropoff_lon))
                        total_duration = int(self._eta_seconds(pickup_pos, dropoff_pos))
                except Exception:
                    eta_to_pickup = None

            # Après pickup : ETA chauffeur → destination (dropoff)
            if in_progress and dropoff_lat is not None and dropoff_lon is not None:
                try:
                    dropoff_pos = (float(dropoff_lat), float(dropoff_lon))
                    eta_to_dropoff = int(self._eta_seconds(driver_pos, dropoff_pos))
                except Exception:
                    eta_to_dropoff = None
            else:
                eta_to_dropoff = None

            estimated_arrival = (
                (current_time + timedelta(seconds=eta_to_pickup)).isoformat()
                if eta_to_pickup is not None
                else None
            )
            estimated_arrival_dropoff = (
                (current_time + timedelta(seconds=eta_to_dropoff)).isoformat()
                if eta_to_dropoff is not None
                else None
            )

            items.append(
                BookingEtaItem(
                    id=booking_id,
                    eta_to_pickup_seconds=eta_to_pickup,
                    eta_to_dropoff_seconds=eta_to_dropoff,
                    duration_seconds=total_duration,
                    distance_meters=b.distance_meters,
                    estimated_arrival=estimated_arrival,
                    estimated_arrival_dropoff=estimated_arrival_dropoff,
                )
            )

        return DriverBookingsEtaResponse(
            has_gps=True,
            driver_position={"lat": float(driver_lat), "lon": float(driver_lon)},
            bookings=items,
        )
