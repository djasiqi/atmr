from __future__ import annotations

from dataclasses import dataclass

from application.drivers.get_driver_completed_trips import (
    GetDriverCompletedTripsUseCase,
)


@dataclass
class _Trip:
    serialize: dict[str, object]


def test_serializes_trips() -> None:
    def _get(driver_id: int):  # type: ignore[no-untyped-def]
        _ = driver_id
        return [_Trip({"id": 1}), _Trip({"id": 2})]

    uc = GetDriverCompletedTripsUseCase(get_completed_trips_fn=_get)
    res = uc.execute(driver_id=7)
    assert res.status_code == 200
    assert res.response == [{"id": 1}, {"id": 2}]
