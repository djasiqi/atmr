from __future__ import annotations

from dataclasses import dataclass

from application.drivers.get_driver_profile import GetDriverProfileUseCase


@dataclass
class _Driver:
    serialize: dict[str, object]


def test_returns_driver_profile() -> None:
    from application.drivers.get_driver_profile import GetDriverProfileInput

    driver = _Driver(serialize={"id": 1, "name": "X"})
    uc = GetDriverProfileUseCase()
    res = uc.execute(GetDriverProfileInput(driver=driver))
    assert res.status_code == 200
    assert res.response == {"profile": {"id": 1, "name": "X"}}
