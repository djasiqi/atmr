from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from application.drivers.save_driver_push_token import SaveDriverPushTokenUseCase


@dataclass
class _Driver:
    id: int
    push_token: str | None = None


class _UserRepo:
    def __init__(self, user: Any | None):
        self._user = user

    def find_by_public_id(self, public_id: str):  # type: ignore[no-untyped-def]
        _ = public_id
        return self._user


class _DriverRepo:
    def __init__(self, driver: _Driver | None):
        self._driver = driver

    def find_model_by_user_id(self, user_id: int):  # type: ignore[no-untyped-def]
        _ = user_id
        return self._driver

    def find_model_by_id(self, driver_id: int):  # type: ignore[no-untyped-def]
        _ = driver_id
        return self._driver


def test_invalid_token_returns_400() -> None:
    uc = SaveDriverPushTokenUseCase(
        user_repo=_UserRepo(None), driver_repo=_DriverRepo(None)
    )
    res = uc.execute(payload={"token": "x"}, jwt_identity="u")
    assert res.status_code == 400


@patch("application.notifications.upsert_device_token.upsert_device_token")
def test_driver_id_from_payload_sets_token(mock_upsert) -> None:
    driver = _Driver(id=1)
    user = SimpleNamespace(id=123)
    uc = SaveDriverPushTokenUseCase(
        user_repo=_UserRepo(user), driver_repo=_DriverRepo(driver)
    )
    res = uc.execute(
        payload={"token": "x" * 10, "driverId": 1, "device_id": "dev-1"},
        jwt_identity="public_id",
    )
    assert res.status_code == 200
    assert driver.push_token == "x" * 10
    assert res.should_commit is True
    mock_upsert.assert_called_once()


@patch("application.notifications.upsert_device_token.upsert_device_token")
def test_fallback_jwt_resolves_driver(mock_upsert) -> None:
    driver = _Driver(id=7)
    user = SimpleNamespace(id=123)
    uc = SaveDriverPushTokenUseCase(
        user_repo=_UserRepo(user), driver_repo=_DriverRepo(driver)
    )
    res = uc.execute(
        payload={"token": "x" * 10, "device_id": "dev-7"},
        jwt_identity="public_id",
    )
    assert res.status_code == 200
    assert driver.push_token == "x" * 10
    mock_upsert.assert_called_once()
