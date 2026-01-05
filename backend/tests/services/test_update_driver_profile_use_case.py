from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from application.drivers.update_driver_profile import UpdateDriverProfileUseCase


@dataclass
class _User:
    first_name: str | None = None
    last_name: str | None = None
    phone: str | None = None


@dataclass
class _Driver:
    user: _User | None = None
    is_active: bool = False
    contract_type: str | None = None
    weekly_hours: int | None = None
    hourly_rate_cents: int | None = None
    employment_start_date: Any | None = None
    employment_end_date: Any | None = None
    license_valid_until: Any | None = None
    medical_valid_until: Any | None = None
    license_categories: list[str] = field(default_factory=list)
    trainings: Any | None = None
    serialize: dict[str, object] = field(default_factory=lambda: {"id": 1})


def test_missing_user_returns_500() -> None:
    driver = _Driver(user=None)
    uc = UpdateDriverProfileUseCase()
    res = uc.execute(driver=driver, validated_data={})
    assert res.status_code == 500
    assert res.should_commit is False


def test_updates_user_fields_and_sets_active() -> None:
    driver = _Driver(user=_User(), is_active=False)
    uc = UpdateDriverProfileUseCase()
    res = uc.execute(
        driver=driver,
        validated_data={
            "first_name": "A",
            "last_name": "B",
            "phone": "123",
            "status": "disponible",
            "contract_type": "cdi",
            "weekly_hours": 35,
            "hourly_rate_cents": 1200,
            "license_categories": ["B", "C1"],
            "trainings": [{"name": "t"}],
        },
    )
    assert res.status_code == 200
    assert res.should_commit is True
    assert driver.user is not None
    assert driver.user.first_name == "A"
    assert driver.user.last_name == "B"
    assert driver.user.phone == "123"
    assert driver.is_active is True
    assert driver.contract_type == "CDI"
    assert driver.weekly_hours == 35
    assert driver.hourly_rate_cents == 1200
    assert driver.license_categories == ["B", "C1"]
