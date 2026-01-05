from __future__ import annotations

from dataclasses import dataclass

from application.drivers.update_driver_availability import (
    UpdateDriverAvailabilityUseCase,
)


@dataclass
class _Driver:
    is_available: bool = False


def test_missing_payload_returns_400() -> None:
    driver = _Driver(is_available=False)
    uc = UpdateDriverAvailabilityUseCase()
    res = uc.execute(driver=driver, payload=None)
    assert res.status_code == 400
    assert res.should_commit is False
    assert driver.is_available is False


def test_missing_is_available_returns_400() -> None:
    driver = _Driver(is_available=False)
    uc = UpdateDriverAvailabilityUseCase()
    res = uc.execute(driver=driver, payload={})
    assert res.status_code == 400
    assert res.should_commit is False
    assert driver.is_available is False


def test_true_sets_available_and_requires_commit() -> None:
    driver = _Driver(is_available=False)
    uc = UpdateDriverAvailabilityUseCase()
    res = uc.execute(driver=driver, payload={"is_available": True})
    assert res.status_code == 200
    assert res.should_commit is True
    assert driver.is_available is True


def test_string_false_sets_unavailable() -> None:
    driver = _Driver(is_available=True)
    uc = UpdateDriverAvailabilityUseCase()
    res = uc.execute(driver=driver, payload={"is_available": "false"})
    assert res.status_code == 200
    assert driver.is_available is False


def test_invalid_string_returns_400() -> None:
    driver = _Driver(is_available=True)
    uc = UpdateDriverAvailabilityUseCase()
    res = uc.execute(driver=driver, payload={"is_available": "nope"})
    assert res.status_code == 400
    assert res.should_commit is False
    assert driver.is_available is True
