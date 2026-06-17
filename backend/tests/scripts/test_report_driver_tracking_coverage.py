from __future__ import annotations

from datetime import UTC, datetime, timedelta

from scripts.report_driver_tracking_coverage import (
    ROOT_CAUSE_VALUES,
    _infer_root_cause,
)


def test_infer_root_cause_tracking_ok() -> None:
    cause, _plan = _infer_root_cause(
        in_pipeline=True,
        push_token_present=True,
        health={},
        last_gps_at=datetime.now(UTC),
        now=datetime.now(UTC),
    )
    assert cause == "tracking_ok"


def test_infer_root_cause_no_push_token() -> None:
    cause, plan = _infer_root_cause(
        in_pipeline=False,
        push_token_present=False,
        health={},
        last_gps_at=None,
        now=datetime.now(UTC),
    )
    assert cause == "no_push_token"
    assert "push" in plan.lower()


def test_infer_root_cause_fgs_android() -> None:
    cause, _plan = _infer_root_cause(
        in_pipeline=False,
        push_token_present=True,
        health={"platform": "android", "foreground_service_running": False},
        last_gps_at=datetime.now(UTC) - timedelta(days=2),
        now=datetime.now(UTC),
    )
    assert cause == "fgs_not_running"


def test_root_cause_taxonomy_complete() -> None:
    assert "tracking_ok" in ROOT_CAUSE_VALUES
    assert "investigation_required" in ROOT_CAUSE_VALUES
