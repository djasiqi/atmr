from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Protocol, Sequence


@dataclass(frozen=True, slots=True)
class BookingEtaItem:
    id: int
    eta_to_pickup_seconds: int | None
    duration_seconds: int | None
    distance_meters: float | int | None
    estimated_arrival: str | None


@dataclass(frozen=True, slots=True)
class DriverBookingsEtaResponse:
    has_gps: bool
    driver_position: dict[str, float] | None
    bookings: list[BookingEtaItem]


class _BookingLike(Protocol):
    id: int
    pickup_lat: float | None
    pickup_lon: float | None
    dropoff_lat: float | None
    dropoff_lon: float | None
    duration_seconds: int | None
    distance_meters: float | int | None
    status: object | None


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
        bookings: Sequence[_BookingLike],
    ) -> DriverBookingsEtaResponse:
        if driver_lat is None or driver_lon is None:
            return DriverBookingsEtaResponse(
                has_gps=False,
                driver_position=None,
                bookings=[
                    BookingEtaItem(
                        id=int(b.id),
                        eta_to_pickup_seconds=None,
                        duration_seconds=b.duration_seconds,
                        distance_meters=b.distance_meters,
                        estimated_arrival=None,
                    )
                    for b in bookings
                ],
            )

        driver_pos = (float(driver_lat), float(driver_lon))
        current_time = self._now_local()

        items: list[BookingEtaItem] = []
        for b in bookings:
            booking_id = int(b.id)
            pickup_lat = b.pickup_lat
            pickup_lon = b.pickup_lon
            dropoff_lat = b.dropoff_lat
            dropoff_lon = b.dropoff_lon
            status = b.status

            eta_to_pickup: int | None = None
            total_duration = b.duration_seconds

            if pickup_lat is not None and pickup_lon is not None:
                try:
                    pickup_pos = (float(pickup_lat), float(pickup_lon))
                    eta_to_pickup = int(self._eta_seconds(driver_pos, pickup_pos))

                    # Recalculer la durée totale si on a dropoff + booking pas "in progress"
                    if (
                        dropoff_lat is not None
                        and dropoff_lon is not None
                        and str(status) != "BookingStatus.IN_PROGRESS"
                        and str(getattr(status, "value", status)) != "in_progress"
                    ):
                        dropoff_pos = (float(dropoff_lat), float(dropoff_lon))
                        total_duration = int(self._eta_seconds(pickup_pos, dropoff_pos))
                except Exception:
                    eta_to_pickup = None

            estimated_arrival = (
                (current_time + timedelta(seconds=eta_to_pickup)).isoformat()
                if eta_to_pickup is not None
                else None
            )

            items.append(
                BookingEtaItem(
                    id=booking_id,
                    eta_to_pickup_seconds=eta_to_pickup,
                    duration_seconds=total_duration,
                    distance_meters=b.distance_meters,
                    estimated_arrival=estimated_arrival,
                )
            )

        return DriverBookingsEtaResponse(
            has_gps=True,
            driver_position={"lat": float(driver_lat), "lon": float(driver_lon)},
            bookings=items,
        )
