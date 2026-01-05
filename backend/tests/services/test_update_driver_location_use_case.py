from __future__ import annotations

from dataclasses import dataclass

from application.drivers.update_driver_location import (
    UpdateDriverLocationCommand,
    UpdateDriverLocationUseCase,
)


@dataclass
class _FakeRes:
    snapped_lat: float
    snapped_lon: float
    source: str
    geofence_events: list[str]


def test_update_driver_location_use_case_returns_snapped_and_events() -> None:
    def fake_update_location(**_kwargs):  # type: ignore[no-untyped-def]
        return _FakeRes(
            snapped_lat=1.1,
            snapped_lon=2.2,
            source="osrm_nearest",
            geofence_events=["arrived_at_pickup"],
        )

    uc = UpdateDriverLocationUseCase(update_location_fn=fake_update_location)
    res = uc.execute(
        UpdateDriverLocationCommand(driver_id=1, latitude=1.0, longitude=2.0)
    )
    assert res.snapped_lat == 1.1
    assert res.snapped_lon == 2.2
    assert res.source == "osrm_nearest"
    assert res.geofence_events == ["arrived_at_pickup"]
