from __future__ import annotations

from typing import Any

from application.dispatch.dispatch_use_case import DispatchUseCase
from domain.dispatch.commands import DispatchRunRequestCommand


def test_dispatch_use_case_prepare_params() -> None:
    uc = DispatchUseCase(
        get_bookings_for_day_fn=lambda _cid, _d: [],
        getenv_fn=lambda _k, _d: _d,
        engine_run_fn=lambda **_kw: {},
        validate_assignments_fn=lambda _a, **_kw: {
            "valid": True,
            "errors": [],
            "warnings": [],
        },
    )

    params = uc.prepare_dispatch_params(
        validated_data={
            "for_date": "2025-01-15",
            "regular_first": False,
            "allow_emergency": True,
            "overrides": {"x": 1},
        },
        company_id=123,
        effective_mode="auto",
    )

    assert params["company_id"] == 123
    assert params["for_date"] == "2025-01-15"
    assert params["mode"] == "auto"
    assert params["regular_first"] is False
    assert params["allow_emergency"] is True
    assert params["overrides"] == {"x": 1}


def test_dispatch_use_case_execute_dispatch_sync_calls_engine_and_validation() -> None:
    calls: dict[str, Any] = {"engine": 0, "validate": 0}

    def fake_engine_run(**params):  # type: ignore[no-untyped-def]
        calls["engine"] += 1
        assert params["company_id"] == 1
        return {"assignments": [{"booking_id": 1, "driver_id": 2}]}

    def fake_validate(assignments, strict=False):  # type: ignore[no-untyped-def]
        _ = strict
        calls["validate"] += 1
        assert assignments
        return {"valid": True, "errors": [], "warnings": []}

    uc = DispatchUseCase(
        get_bookings_for_day_fn=lambda _cid, _d: [],
        getenv_fn=lambda _k, _d: _d,
        engine_run_fn=fake_engine_run,
        validate_assignments_fn=fake_validate,
    )

    result, validation_info = uc.execute_dispatch_sync(
        {"company_id": 1, "for_date": "2025-01-15"}
    )
    assert result["assignments"]
    assert validation_info is None
    assert calls["engine"] == 1
    assert calls["validate"] == 1


def test_dispatch_use_case_validate_and_normalize_rejects_unknown_override_keys() -> (
    None
):
    uc = DispatchUseCase(
        get_bookings_for_day_fn=lambda _cid, _d: [],
        getenv_fn=lambda _k, _d: _d,
        engine_run_fn=lambda **_kw: {},
        validate_assignments_fn=lambda _a, **_kw: {
            "valid": True,
            "errors": [],
            "warnings": [],
        },
    )

    validated, error_response, status = uc.validate_and_normalize_request(
        DispatchRunRequestCommand(
            company_id=1,
            body={"for_date": "2025-01-15", "overrides": {"unknown_key": {"a": 1}}},
        )
    )
    assert validated is None
    assert status == 400
    assert error_response is not None
