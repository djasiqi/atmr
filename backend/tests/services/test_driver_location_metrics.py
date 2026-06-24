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


def test_inc_tracking_delivery_result_labels() -> None:
    if m._TRACKING_DELIVERY_RESULT is None:
        pytest.skip("prometheus_client Counter unavailable")
    inc = MagicMock()
    with patch.object(m, "_TRACKING_DELIVERY_RESULT") as mock_c:
        mock_c.labels.return_value = MagicMock(inc=inc)
        m.inc_tracking_delivery_result(
            mode="availability_presence",
            transport="http",
            result="success",
        )
    mock_c.labels.assert_called_once_with(
        mode="availability_presence", transport="http", result="success"
    )
    inc.assert_called_once()


def test_inc_batch_rate_limited() -> None:
    if m._BATCH_RATE_LIMITED is None:
        pytest.skip("prometheus_client Counter unavailable")
    with patch.object(m, "_BATCH_RATE_LIMITED") as mock_counter:
        mock_counter.inc = MagicMock()
        m.inc_batch_rate_limited()
    mock_counter.inc.assert_called_once()


def test_inc_tracking_mission_live_missing_mission_id() -> None:
    if m._TRACKING_MISSION_LIVE_MISSING_MISSION_ID is None:
        pytest.skip("prometheus_client Counter unavailable")
    inc = MagicMock()
    with patch.object(m, "_TRACKING_MISSION_LIVE_MISSING_MISSION_ID") as mock_counter:
        mock_counter.labels.return_value = MagicMock(inc=inc)
        m.inc_tracking_mission_live_missing_mission_id(
            transport="http",
            action="downgraded",
        )
    mock_counter.labels.assert_called_once_with(transport="http", action="downgraded")
    inc.assert_called_once()


def test_observe_gps_quality_accuracy() -> None:
    if m._GPS_ACCURACY is None:
        pytest.skip("prometheus_client Histogram unavailable")
    observe = MagicMock()
    with patch.object(m, "_GPS_ACCURACY") as mock_hist:
        mock_hist.labels.return_value = MagicMock(observe=observe)
        m.observe_gps_quality(
            platform="ios",
            location_mode="mission_live",
            transport="socket_batch",
            accuracy=12.5,
        )
    observe.assert_called_once_with(12.5)


def test_inc_tracking_id_propagated() -> None:
    if m._TRACKING_ID_PROPAGATED is None:
        pytest.skip("prometheus_client Counter unavailable")
    inc = MagicMock()
    with patch.object(m, "_TRACKING_ID_PROPAGATED") as mock_counter:
        mock_counter.labels.return_value = MagicMock(inc=inc)
        m.inc_tracking_id_propagated(transport="socket_batch", propagated=True)
    inc.assert_called_once()


def test_inc_received_accepts_kafka_transport() -> None:
    if m._RECEIVED is None:
        pytest.skip("prometheus_client Counter unavailable")
    inc = MagicMock()
    with patch.object(m, "_RECEIVED") as mock_r, patch.object(m, "_INGESTED") as mock_i:
        mock_r.labels.return_value = MagicMock(inc=inc)
        mock_i.labels.return_value = MagicMock(inc=MagicMock())
        m.inc_received(transport="kafka", location_mode="mission_live")
    mock_r.labels.assert_called_once_with(transport="kafka", location_mode="mission_live")


def test_inc_tracking_kafka_dlq_force_commit() -> None:
    if m._TRACKING_KAFKA_DLQ_FORCE_COMMIT is None:
        pytest.skip("prometheus_client Counter unavailable")
    inc = MagicMock()
    with patch.object(m, "_TRACKING_KAFKA_DLQ_FORCE_COMMIT") as mock_c:
        mock_c.labels.return_value = MagicMock(inc=inc)
        m.inc_tracking_kafka_dlq_force_commit(reason="persist_failed")
    mock_c.labels.assert_called_once_with(reason="persist_failed")
    inc.assert_called_once()


def test_observe_osrm_request_success() -> None:
    if m._TRACKING_OSRM_REQUEST is None or m._TRACKING_OSRM_LATENCY is None:
        pytest.skip("prometheus_client metrics unavailable")
    inc = MagicMock()
    observe = MagicMock()
    with patch.object(m, "_TRACKING_OSRM_REQUEST") as mock_c, patch.object(m, "_TRACKING_OSRM_LATENCY") as mock_h:
        mock_c.labels.return_value = MagicMock(inc=inc)
        mock_h.labels.return_value = MagicMock(observe=observe)
        m.observe_osrm_request(operation="nearest", result="success", duration_sec=0.42)
    inc.assert_called_once()
    observe.assert_called_once_with(0.42)


def test_inc_tracking_kafka_persist_maps_unknown_status_to_failed() -> None:
    if m._TRACKING_KAFKA_PERSIST is None:
        pytest.skip("prometheus_client Counter unavailable")
    inc = MagicMock()
    with patch.object(m, "_TRACKING_KAFKA_PERSIST") as mock_counter:
        mock_counter.labels.return_value = MagicMock(inc=inc)
        m.inc_tracking_kafka_persist(accept_status="weird_status")
    mock_counter.labels.assert_called_once_with(accept_status="failed")
    inc.assert_called_once()


def test_set_tracking_kafka_consumer_lag_sets_gauge() -> None:
    if m._TRACKING_KAFKA_CONSUMER_LAG is None:
        pytest.skip("prometheus_client Gauge unavailable")
    setter = MagicMock()
    with patch.object(m, "_TRACKING_KAFKA_CONSUMER_LAG") as mock_g:
        mock_g.labels.return_value = MagicMock(set=setter)
        m.set_tracking_kafka_consumer_lag(
            group="tracking-ingest-consumer-group",
            topic="driver.location.raw.v2",
            partition=3,
            lag=42,
        )
    mock_g.labels.assert_called_once_with(
        group="tracking-ingest-consumer-group",
        topic="driver.location.raw.v2",
        partition="3",
    )
    setter.assert_called_once_with(42.0)


def test_set_tracking_kafka_consumer_lag_clamps_negative() -> None:
    if m._TRACKING_KAFKA_CONSUMER_LAG is None:
        pytest.skip("prometheus_client Gauge unavailable")
    setter = MagicMock()
    with patch.object(m, "_TRACKING_KAFKA_CONSUMER_LAG") as mock_g:
        mock_g.labels.return_value = MagicMock(set=setter)
        m.set_tracking_kafka_consumer_lag(
            group="g", topic="t", partition=0, lag=-5
        )
    setter.assert_called_once_with(0.0)


def test_set_tracking_kafka_consumer_lag_noop_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DRIVER_LOCATION_METRICS_ENABLED", "false")
    with patch.object(m, "_TRACKING_KAFKA_CONSUMER_LAG") as mock_g:
        m.set_tracking_kafka_consumer_lag(group="g", topic="t", partition=1, lag=10)
    mock_g.labels.assert_not_called()
