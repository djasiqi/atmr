from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from application.drivers.get_driver_bookings_eta import GetDriverBookingsETAUseCase


@dataclass
class _Booking:
    id: int
    pickup_lat: float | None = None
    pickup_lon: float | None = None
    dropoff_lat: float | None = None
    dropoff_lon: float | None = None
    duration_seconds: int | None = None
    distance_meters: float | None = None
    status: object | None = None


def test_eta_use_case_without_gps_returns_static_fields() -> None:
    uc = GetDriverBookingsETAUseCase(
        eta_seconds_fn=lambda _a, _b: 42,
        now_local_fn=lambda: datetime(2025, 12, 12, 10, 0, 0),
    )
    resp = uc.execute(
        driver_lat=None,
        driver_lon=None,
        bookings=[_Booking(id=1, duration_seconds=100, distance_meters=12.0)],
    )
    assert resp.has_gps is False
    assert resp.driver_position is None
    assert resp.bookings[0].id == 1
    assert resp.bookings[0].eta_to_pickup_seconds is None


def test_eta_use_case_with_gps_computes_eta_and_estimated_arrival() -> None:
    now = datetime(2025, 12, 12, 10, 0, 0)

    def eta_seconds(_a, _b):  # type: ignore[no-untyped-def]
        return 60

    uc = GetDriverBookingsETAUseCase(
        eta_seconds_fn=eta_seconds,
        now_local_fn=lambda: now,
    )
    resp = uc.execute(
        driver_lat=1.0,
        driver_lon=2.0,
        bookings=[
            _Booking(
                id=1,
                pickup_lat=1.1,
                pickup_lon=2.2,
                duration_seconds=999,
                distance_meters=12.0,
            )
        ],
    )
    assert resp.has_gps is True
    assert resp.bookings[0].eta_to_pickup_seconds == 60
    assert (
        resp.bookings[0].estimated_arrival == (now + timedelta(seconds=60)).isoformat()
    )
