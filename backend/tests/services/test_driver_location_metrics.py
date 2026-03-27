"""Tests métriques PR1 localisation (skew horloge)."""

from __future__ import annotations

import os

import pytest

from services.monitoring import driver_location_metrics as m


def test_observe_clock_skew_seconds_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DRIVER_LOCATION_METRICS_ENABLED", "false")
    m.observe_clock_skew_seconds(location_mode="mission_live", skew_seconds=12.5)


def test_observe_clock_skew_seconds_skips_out_of_range() -> None:
    if m._CLOCK_SKEW is None:
        pytest.skip("prometheus_client Histogram unavailable")
    m.observe_clock_skew_seconds(location_mode="mission_live", skew_seconds=-1.0)
    m.observe_clock_skew_seconds(location_mode="mission_live", skew_seconds=999999.0)


def test_observe_clock_skew_seconds_observes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DRIVER_LOCATION_METRICS_ENABLED", raising=False)
    if m._CLOCK_SKEW is None:
        pytest.skip("prometheus_client Histogram unavailable")
    m.observe_clock_skew_seconds(location_mode="availability_presence", skew_seconds=15.0)
