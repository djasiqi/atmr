from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from application.drivers.update_driver_location import UpdateDriverLocationResult
from services.tracking.ingest_consumer import TrackingIngestConsumer
from services.tracking.ingest_persist import persist_driver_location_from_kafka


def _record():
    return SimpleNamespace(
        topic="raw",
        partition=0,
        offset=42,
        key="driver_7",
        timestamp=1_700_000_000_000,
        value=None,
    )


def test_process_record_persists_when_flag_enabled(monkeypatch):
    consumer = TrackingIngestConsumer()
    message = {
        "driver_id": 7,
        "company_id": 99,
        "location_event_id": "ev-1",
        "received_at_ms": 1_700_000_000_000,
        "payload": {
            "latitude": 46.2,
            "longitude": 6.1,
            "recorded_at": "2026-06-23T10:00:00+00:00",
            "location_mode": "mission_live",
        },
    }
    record = _record()
    record.value = message

    uc_result = UpdateDriverLocationResult(
        snapped_lat=46.2001,
        snapped_lon=6.1001,
        source="osrm",
        geofence_events=[],
        accept_status="accepted_canonical",
        accept_reason="",
        received_at="2026-06-23T10:00:01+00:00",
    )
    enriched = {
        **message,
        "persist_result": {
            "accept_status": "accepted_canonical",
            "accept_reason": "",
            "snapped_lat": 46.2001,
            "snapped_lon": 6.1001,
            "dedup_skipped": False,
        },
    }

    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_INGEST_PERSIST_ENABLED", True
    )
    monkeypatch.setattr(consumer, "_publish_with_ack", MagicMock())
    monkeypatch.setattr(consumer, "_commit_current", MagicMock())
    monkeypatch.setattr(consumer, "_observe_e2e_latency", MagicMock())

    with patch(
        "services.tracking.ingest_persist.persist_driver_location_from_kafka",
        return_value=(enriched, uc_result),
    ) as mock_persist:
        ok = consumer._process_record(record)

    assert ok is True
    mock_persist.assert_called_once()
    publish_kwargs = consumer._publish_with_ack.call_args.kwargs
    assert publish_kwargs["message"]["persist_result"]["accept_status"] == (
        "accepted_canonical"
    )


def test_process_record_skips_persist_when_flag_disabled(monkeypatch):
    consumer = TrackingIngestConsumer()
    message = {
        "driver_id": 7,
        "payload": {"latitude": 46.2, "longitude": 6.1},
        "received_at_ms": 1_700_000_000_000,
    }
    record = _record()
    record.value = message

    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_INGEST_PERSIST_ENABLED", False
    )
    monkeypatch.setattr(consumer, "_publish_with_ack", MagicMock())
    monkeypatch.setattr(consumer, "_commit_current", MagicMock())
    monkeypatch.setattr(consumer, "_observe_e2e_latency", MagicMock())

    with patch(
        "services.tracking.ingest_persist.persist_driver_location_from_kafka",
    ) as mock_persist:
        ok = consumer._process_record(record)

    assert ok is True
    mock_persist.assert_not_called()


def test_persist_driver_location_logs_correlation_for_kafka(monkeypatch):
    uc_result = UpdateDriverLocationResult(
        snapped_lat=46.2001,
        snapped_lon=6.1001,
        source="osrm",
        geofence_events=[],
        accept_status="accepted_canonical",
        accept_reason="",
        received_at="2026-06-23T10:00:01+00:00",
    )
    mock_uc = MagicMock()
    mock_uc.execute.return_value = uc_result
    mock_loc_svc = MagicMock()
    mock_loc_svc.resolve_normalized_location_mode.return_value = "mission_live"

    mock_app = MagicMock()
    mock_ctx = MagicMock()
    mock_app.app_context.return_value = mock_ctx
    mock_ctx.__enter__ = MagicMock(return_value=None)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch("celery_app.get_flask_app", return_value=mock_app),
        patch(
            "services.tracking.ingest_persist.UpdateDriverLocationUseCase",
            return_value=mock_uc,
        ),
        patch(
            "services.tracking.ingest_persist.create_location_update_fn",
            return_value=MagicMock(),
        ),
        patch(
            "services.geolocation.location.get_location_service",
            return_value=mock_loc_svc,
        ),
        patch(
            "services.monitoring.location_correlation_log.log_driver_location_processed",
        ) as mock_log,
    ):
        enriched, result = persist_driver_location_from_kafka(
            {
                "driver_id": 7,
                "company_id": 99,
                "location_event_id": "ev-1",
                "payload": {
                    "latitude": 46.2,
                    "longitude": 6.1,
                    "recorded_at": "2026-06-23T10:00:00+00:00",
                    "location_mode": "mission_live",
                },
            },
            driver_id=7,
        )

    assert result.accept_status == "accepted_canonical"
    assert enriched["persist_result"]["accept_status"] == "accepted_canonical"
    mock_log.assert_called_once_with(
        driver_id=7,
        company_id=99,
        transport="kafka",
        location_mode="mission_live",
        accept_status="accepted_canonical",
        accept_reason="",
        location_event_id="ev-1",
    )
