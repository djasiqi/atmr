from __future__ import annotations

from application.dispatch.dispatch_use_case import DispatchUseCase


def test_dispatch_use_case_should_force_async_uses_injected_deps() -> None:
    def fake_get_bookings_for_day(company_id: int, for_date: str):
        assert company_id == 123
        assert for_date == "2025-01-15"
        return [object()] * 11

    def fake_getenv(key: str, default: str) -> str:
        assert key == "DISPATCH_SYNC_MAX_BOOKINGS"
        assert default == "10"
        return "10"

    use_case = DispatchUseCase(
        get_bookings_for_day_fn=fake_get_bookings_for_day,
        getenv_fn=fake_getenv,
    )

    should_force, reason = use_case.should_force_async_mode(
        company_id=123, for_date="2025-01-15", is_async=False, getenv_fn=fake_getenv
    )
    assert should_force is True
    assert reason is not None


def test_dispatch_use_case_should_not_force_async_when_already_async() -> None:
    use_case = DispatchUseCase(
        get_bookings_for_day_fn=lambda _cid, _date: [object()] * 999,
        getenv_fn=lambda _k, _d: "0",
    )

    should_force, reason = use_case.should_force_async_mode(
        company_id=1,
        for_date="2025-01-15",
        is_async=True,
        getenv_fn=lambda _k, _d: "0",
    )
    assert should_force is False
    assert reason is None
