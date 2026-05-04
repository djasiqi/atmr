"""Tests métriques PR1 localisation (skew horloge)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from services.monitoring import driver_location_metrics as m


def test_observe_clock_skew_seconds_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    m.observe_clock_skew_seconds(
        location_mode="availability_presence", skew_seconds=15.0
    )


def test_inc_received_increments_received_and_ingested_same_labels() -> None:
    """Patch C : inc_received doit incrémenter received et ingested avec les mêmes labels."""
    if m._RECEIVED is None or m._INGESTED is None:
        pytest.skip("prometheus_client Counter unavailable")
    r_inc = MagicMock()
    i_inc = MagicMock()
    # Ne pas patcher Counter.labels directement (impl prometheus_client) : remplacer les compteurs.
    with patch.object(m, "_RECEIVED") as mock_r, patch.object(m, "_INGESTED") as mock_i:
        mock_r.labels.return_value = MagicMock(inc=r_inc)
        mock_i.labels.return_value = MagicMock(inc=i_inc)
        m.inc_received(transport="socket_batch", location_mode="mission_live")
    mock_r.labels.assert_called_once_with(
        transport="socket_batch", location_mode="mission_live"
    )
    mock_i.labels.assert_called_once_with(
        transport="socket_batch", location_mode="mission_live"
    )
    r_inc.assert_called_once()
    i_inc.assert_called_once()
