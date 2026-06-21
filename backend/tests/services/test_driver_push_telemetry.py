"""Tests service télémétrie push chauffeur."""

from __future__ import annotations

from services.monitoring.driver_push_telemetry import ingest_driver_push_telemetry


def test_ingest_driver_push_telemetry_success(caplog):
    with caplog.at_level("INFO"):
        result = ingest_driver_push_telemetry(
            driver_id=7514,
            body={
                "event": "driver_push.disclosure_blocked",
                "platform": "android",
                "source": "driver.notifications.bridge",
            },
        )
    assert result["ok"] is True
    assert result["event"] == "driver_push.disclosure_blocked"
    assert any("driver_push_telemetry event=driver_push.disclosure_blocked" in rec.message for rec in caplog.records)


def test_ingest_driver_push_telemetry_token_acquired(caplog):
    with caplog.at_level("INFO"):
        result = ingest_driver_push_telemetry(
            driver_id=7514,
            body={
                "event": "driver_push.token_acquired",
                "platform": "android",
                "source": "driver.notifications.bridge",
                "provider": "fcm",
                "token_length": 163,
            },
        )
    assert result["ok"] is True
    assert any("token_length=163" in rec.message for rec in caplog.records)
    result = ingest_driver_push_telemetry(
        driver_id=7514,
        body={"event": "driver_push.unknown", "platform": "android"},
    )
    assert result["ok"] is False
    assert result["error"] == "unknown_event"
