from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from application.drivers.get_trip_tracking_replay import GetTripTrackingReplayUseCase


@dataclass
class _Assignment:
    driver_id: int
    booking_id: int


@dataclass
class _Pos:
    latitude: float
    longitude: float
    speed: float | None
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "speed": self.speed,
            "timestamp": self.timestamp.isoformat(),
        }


def test_unauthorized_returns_404() -> None:
    uc = GetTripTrackingReplayUseCase(
        get_assignment_fn=lambda _aid: _Assignment(driver_id=2, booking_id=9),
        get_positions_fn=lambda _aid: [],
        haversine_distance_fn=lambda *_a: 0.0,
    )
    res = uc.execute(assignment_id=1, driver_id=1)
    assert res.status_code == 404


def test_empty_positions_returns_zero_analytics() -> None:
    uc = GetTripTrackingReplayUseCase(
        get_assignment_fn=lambda _aid: _Assignment(driver_id=1, booking_id=9),
        get_positions_fn=lambda _aid: [],
        haversine_distance_fn=lambda *_a: 0.0,
    )
    res = uc.execute(assignment_id=1, driver_id=1)
    assert res.status_code == 200
    assert res.response["positions"] == []
    assert res.response["analytics"]["total_positions"] == 0


def test_positions_compute_distance_duration_and_speeds() -> None:
    t0 = datetime(2025, 1, 1, 10, 0, 0)
    positions = [
        _Pos(0.0, 0.0, 2.0, t0),  # 7.2 km/h
        _Pos(0.0, 1.0, 0.5, t0 + timedelta(seconds=10)),  # stop (<1 m/s) + 1.8 km/h
        _Pos(0.0, 2.0, None, t0 + timedelta(seconds=20)),
    ]

    def _haversine(_a, _b, _c, _d):  # type: ignore[no-untyped-def]
        return 1.0  # 1 km par segment

    uc = GetTripTrackingReplayUseCase(
        get_assignment_fn=lambda _aid: _Assignment(driver_id=1, booking_id=9),
        get_positions_fn=lambda _aid: positions,
        haversine_distance_fn=_haversine,
    )
    res = uc.execute(assignment_id=1, driver_id=1)
    assert res.status_code == 200
    assert res.response["analytics"]["total_positions"] == 3
    assert res.response["analytics"]["duration_seconds"] == 20
    assert res.response["analytics"]["total_distance_km"] == 2.0
    assert res.response["analytics"]["stops_count"] == 1
    assert res.response["analytics"]["max_speed_kmh"] == 7.2
