"""E2E pipeline tracking — inject GPS fictif (N4)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_pipeline_enqueue_returns_trace_id() -> None:
    with patch("services.tracking.ingest_producer.tracking_ingest_producer") as mock_prod:
        mock_prod.enqueue.return_value = {
            "queued": True,
            "trace_id": "tr-test-1",
            "topic": "driver.location.raw.v2",
        }
        from services.tracking import enqueue_tracking_event

        result = enqueue_tracking_event(
            driver_id=7514,
            payload={
                "latitude": 46.2,
                "longitude": 6.1,
                "location_event_id": "trk_test_e2e_1",
                "recorded_at": "2026-06-25T12:00:00+00:00",
            },
            source="http",
            company_id=1,
        )
        assert result.get("trace_id") == "tr-test-1"
        mock_prod.enqueue.assert_called_once()


def test_invariant_violation_metric_callable() -> None:
    from services.monitoring.driver_location_metrics import inc_tracking_invariant_violation

    inc_tracking_invariant_violation(
        invariant_id="INV-1",
        company_id=1,
        driver_id=7514,
    )
