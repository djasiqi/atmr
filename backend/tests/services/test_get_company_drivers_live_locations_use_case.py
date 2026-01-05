from __future__ import annotations

from dataclasses import dataclass

from application.drivers.get_company_drivers_live_locations import (
    GetCompanyDriversLiveLocationsUseCase,
)


@dataclass
class _Driver:
    id: int


class _DriverRepo:
    def __init__(self, drivers: list[_Driver]) -> None:
        self._drivers = drivers

    def find_models_by_company_id(self, company_id: int):  # type: ignore[no-untyped-def]
        _ = company_id
        return self._drivers


def test_returns_only_drivers_with_locations() -> None:
    repo = _DriverRepo([_Driver(id=1), _Driver(id=2)])

    def _store(driver_id: int):  # type: ignore[no-untyped-def]
        if driver_id == 1:
            return {"lat": 1.0, "lon": 2.0}
        return None

    uc = GetCompanyDriversLiveLocationsUseCase(
        driver_repo=repo, get_last_location_fn=_store
    )
    res = uc.execute(company_id=10)
    assert res.status_code == 200
    assert res.response == {"items": [{"driver_id": 1, "lat": 1.0, "lon": 2.0}]}
