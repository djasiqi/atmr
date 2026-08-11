from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from application.drivers.update_driver_location import (
    UpdateDriverLocationCommand,
    UpdateDriverLocationUseCase,
)


@dataclass
class _FakeRes:
    snapped_lat: float
    snapped_lon: float
    source: str
    geofence_events: list[str]
    accept_status: str = "accepted_canonical"
    accept_reason: str = ""
    received_at: str | None = None


def test_update_driver_location_use_case_returns_snapped_and_events() -> None:
    def fake_update_location(**_kwargs):  # type: ignore[no-untyped-def]
        return _FakeRes(
            snapped_lat=1.1,
            snapped_lon=2.2,
            source="osrm_nearest",
            geofence_events=["arrived_at_pickup"],
        )

    with patch(
        "application.drivers.update_driver_location.should_skip_location_ingest",
        return_value=(False, None),
    ):
        uc = UpdateDriverLocationUseCase(update_location_fn=fake_update_location)
        res = uc.execute(
            UpdateDriverLocationCommand(driver_id=1, latitude=1.0, longitude=2.0)
        )
    assert res.snapped_lat == 1.1
    assert res.snapped_lon == 2.2
    assert res.source == "osrm_nearest"
    assert res.geofence_events == ["arrived_at_pickup"]
    assert res.accept_status == "accepted_canonical"


def test_update_driver_location_rejects_naive_recorded_at() -> None:
    def fake_update_location(**_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("ne doit pas être appelé")

    uc = UpdateDriverLocationUseCase(update_location_fn=fake_update_location)
    with pytest.raises(ValueError, match="naive"):
        uc.execute(
            UpdateDriverLocationCommand(
                driver_id=1,
                latitude=1.0,
                longitude=2.0,
                recorded_at="2026-08-11T18:00:00",
            )
        )


def test_update_driver_location_accepts_offset_recorded_at() -> None:
    captured: dict = {}

    def fake_update_location(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return _FakeRes(
            snapped_lat=1.0,
            snapped_lon=2.0,
            source="raw",
            geofence_events=[],
        )

    with patch(
        "application.drivers.update_driver_location.should_skip_location_ingest",
        return_value=(False, None),
    ):
        uc = UpdateDriverLocationUseCase(update_location_fn=fake_update_location)
        uc.execute(
            UpdateDriverLocationCommand(
                driver_id=1,
                latitude=1.0,
                longitude=2.0,
                recorded_at="2026-08-11T20:00:00+02:00",
                ts="2026-08-11T20:00:00+02:00",
            )
        )
    assert captured["recorded_at"].tzinfo is not None
    assert captured["recorded_at"].hour == 18
    assert captured["recorded_at"].tzinfo.utcoffset(captured["recorded_at"]).total_seconds() == 0


def test_update_driver_location_use_case_emits_geofence_when_enabled() -> None:
    def fake_update_location(**_kwargs):  # type: ignore[no-untyped-def]
        return _FakeRes(
            snapped_lat=1.1,
            snapped_lon=2.2,
            source="osrm_nearest",
            geofence_events=["arrived_at_pickup"],
        )

    with (
        patch(
            "application.drivers.update_driver_location.should_skip_location_ingest",
            return_value=(False, None),
        ),
        patch(
            "services.tracking.geofence_emit.emit_driver_geofence_events",
        ) as mock_emit,
    ):
        uc = UpdateDriverLocationUseCase(update_location_fn=fake_update_location)
        uc.execute(
            UpdateDriverLocationCommand(
                driver_id=42,
                latitude=1.0,
                longitude=2.0,
                company_id=99,
                emit_geofence=True,
            )
        )

    mock_emit.assert_called_once_with(
        driver_id=42,
        company_id=99,
        geofence_events=["arrived_at_pickup"],
    )


def test_update_driver_location_use_case_skips_geofence_emit_when_disabled() -> None:
    def fake_update_location(**_kwargs):  # type: ignore[no-untyped-def]
        return _FakeRes(
            snapped_lat=1.1,
            snapped_lon=2.2,
            source="osrm_nearest",
            geofence_events=["arrived_at_pickup"],
        )

    with (
        patch(
            "application.drivers.update_driver_location.should_skip_location_ingest",
            return_value=(False, None),
        ),
        patch(
            "services.tracking.geofence_emit.emit_driver_geofence_events",
        ) as mock_emit,
    ):
        uc = UpdateDriverLocationUseCase(update_location_fn=fake_update_location)
        uc.execute(
            UpdateDriverLocationCommand(
                driver_id=42,
                latitude=1.0,
                longitude=2.0,
                company_id=99,
                emit_geofence=False,
            )
        )

    mock_emit.assert_not_called()
