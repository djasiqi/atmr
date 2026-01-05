from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Protocol, Sequence


class _AssignmentLike(Protocol):
    driver_id: int | None
    booking_id: int | None


class _PositionLike(Protocol):
    latitude: float
    longitude: float
    speed: float | None
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class GetTripTrackingReplayResult:
    response: dict[str, Any]
    status_code: int


class GetTripTrackingReplayUseCase:
    """Use-case Application: replay trajet + analytics pour un assignment.

    Règles:
    - assignment doit appartenir au driver (authz)
    - si pas de positions → positions=[], analytics à 0
    """

    def __init__(
        self,
        *,
        get_assignment_fn: Callable[[int], _AssignmentLike | None],
        get_positions_fn: Callable[[int], Sequence[_PositionLike]],
        haversine_distance_fn: Callable[[float, float, float, float], float],
    ) -> None:
        super().__init__()
        self._get_assignment = get_assignment_fn
        self._get_positions = get_positions_fn
        self._haversine_distance = haversine_distance_fn

    def execute(
        self, *, assignment_id: int, driver_id: int
    ) -> GetTripTrackingReplayResult:
        assignment = self._get_assignment(assignment_id)
        if assignment is None or assignment.driver_id != driver_id:
            return GetTripTrackingReplayResult(
                response={"error": "Assignment not found or unauthorized"},
                status_code=404,
            )

        positions = list(self._get_positions(assignment_id))
        if not positions:
            return GetTripTrackingReplayResult(
                response={
                    "assignment_id": assignment_id,
                    "positions": [],
                    "analytics": {
                        "total_positions": 0,
                        "duration_seconds": 0,
                        "average_speed_kmh": 0,
                        "max_speed_kmh": 0,
                        "total_distance_km": 0,
                        "stops_count": 0,
                    },
                },
                status_code=200,
            )

        total_distance_km = 0.0
        speeds: list[float] = []
        stops_count = 0
        last_position: _PositionLike | None = None
        stop_threshold_ms = 1.0  # < 1 m/s = arrêt

        for pos in positions:
            if last_position is not None:
                total_distance_km += self._haversine_distance(
                    last_position.latitude,
                    last_position.longitude,
                    pos.latitude,
                    pos.longitude,
                )
                if pos.speed is not None and pos.speed < stop_threshold_ms:
                    stops_count += 1

            if pos.speed is not None and pos.speed > 0:
                speeds.append(pos.speed * 3.6)  # m/s -> km/h

            last_position = pos

        duration_seconds = (
            (positions[-1].timestamp - positions[0].timestamp).total_seconds()
            if len(positions) > 1
            else 0
        )

        average_speed_kmh = (sum(speeds) / len(speeds)) if speeds else 0.0
        max_speed_kmh = max(speeds) if speeds else 0.0

        return GetTripTrackingReplayResult(
            response={
                "assignment_id": assignment_id,
                "booking_id": assignment.booking_id,
                "positions": [p.to_dict() for p in positions],
                "analytics": {
                    "total_positions": len(positions),
                    "duration_seconds": int(duration_seconds),
                    "average_speed_kmh": round(average_speed_kmh, 2),
                    "max_speed_kmh": round(max_speed_kmh, 2),
                    "total_distance_km": round(total_distance_km, 2),
                    "stops_count": stops_count,
                },
            },
            status_code=200,
        )
