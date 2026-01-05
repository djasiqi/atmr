from __future__ import annotations

from dataclasses import dataclass

from application.drivers.update_driver_admin_profile import (
    UpdateDriverAdminProfileUseCase,
)


@dataclass
class _User:
    phone: str | None = None


@dataclass
class _Driver:
    vehicle_assigned: str | None = None
    brand: str | None = None
    license_plate: str | None = None
    driver_photo: str | None = None
    user: _User | None = None


def test_updates_fields_and_requires_commit() -> None:
    driver = _Driver(user=_User(phone="0"))
    uc = UpdateDriverAdminProfileUseCase()
    res = uc.execute(
        driver=driver,
        payload={
            "vehicle_assigned": "V",
            "brand": "B",
            "license_plate": "LP",
            "photo": "P",
            "phone": "1",
        },
    )
    assert res.status_code == 200
    assert res.should_commit is True
    assert driver.vehicle_assigned == "V"
    assert driver.brand == "B"
    assert driver.license_plate == "LP"
    assert driver.driver_photo == "P"
    assert driver.user is not None
    assert driver.user.phone == "1"
