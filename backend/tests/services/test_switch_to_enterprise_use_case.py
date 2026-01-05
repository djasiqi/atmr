from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from application.drivers.switch_to_enterprise import (
    SwitchToEnterpriseCommand,
    SwitchToEnterpriseUseCase,
)


@dataclass
class _Driver:
    id: int
    driver_type: str
    company_id: int | None
    user_id: int = 1


@dataclass
class _Company:
    id: int
    name: str


@dataclass
class _User:
    id: int
    public_id: str
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None


def test_non_emergency_returns_403() -> None:
    uc = SwitchToEnterpriseUseCase(
        find_company_fn=lambda _cid: None,
        find_company_user_fn=lambda _d, _c: None,
        create_access_token_fn=lambda **_k: "a",
        create_refresh_token_fn=lambda **_k: "r",
        store_refresh_token_fn=None,
        now_utc_fn=lambda: datetime.now(UTC),
        driver_type_emergency="EMERGENCY",
    )
    res = uc.execute(
        SwitchToEnterpriseCommand(
            driver=_Driver(id=1, driver_type="STANDARD", company_id=1),
            access_expires_delta=timedelta(hours=1),
            refresh_expires_delta=timedelta(days=30),
            device_id=None,
            device_name=None,
        )
    )
    assert res.status_code == 403


def test_happy_path_returns_tokens_and_payload() -> None:
    company = _Company(id=10, name="ACME")
    user = _User(id=99, public_id="pub")
    uc = SwitchToEnterpriseUseCase(
        find_company_fn=lambda _cid: company,
        find_company_user_fn=lambda _d, _c: user,
        create_access_token_fn=lambda **_k: "access",
        create_refresh_token_fn=lambda **_k: "refresh",
        store_refresh_token_fn=None,
        now_utc_fn=lambda: datetime(2025, 1, 1, tzinfo=UTC),
        driver_type_emergency="EMERGENCY",
    )
    res = uc.execute(
        SwitchToEnterpriseCommand(
            driver=_Driver(id=1, driver_type="EMERGENCY", company_id=10),
            access_expires_delta=timedelta(hours=1),
            refresh_expires_delta=timedelta(days=30),
            device_id="d",
            device_name="n",
        )
    )
    assert res.status_code == 200
    assert res.response["token"] == "access"
    assert res.response["refresh_token"] == "refresh"
    assert res.response["company"] == {"id": 10, "name": "ACME"}
