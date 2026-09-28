"""METRICS-01 à 03 : Prometheus ne casse pas le push."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


def test_metrics_01_eta_accuracy_rate_registration_is_idempotent() -> None:
    from prometheus_client import Gauge

    from services.monitoring import prometheus as prom

    assert prom.ETA_ACCURACY_RATE is not None
    again = prom._get_or_create_metric(
        Gauge,
        "eta_accuracy_rate",
        "Taux de précision ETA (0-1)",
        ["zone"],
    )
    assert again is prom.ETA_ACCURACY_RATE


def test_metrics_02_track_notification_sent_uses_canonical_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.monitoring import notification_metrics as metrics

    recorded: dict[str, str] = {}

    def _record(
        *, channel: str, notification_type: str, region: str = "unknown"
    ) -> None:
        recorded["channel"] = channel
        recorded["notification_type"] = notification_type
        recorded["region"] = region

    monkeypatch.setattr(
        "services.notifications.metrics.record_notification_sent",
        _record,
    )
    status_labels = MagicMock()
    monkeypatch.setattr(
        metrics,
        "notification_enqueue_status_total",
        MagicMock(labels=MagicMock(return_value=status_labels)),
    )

    metrics.track_notification_sent("booking_assigned", "mission_updates", "queued")

    assert recorded == {
        "channel": "mission_updates",
        "notification_type": "booking_assigned",
        "region": "unknown",
    }
    metrics.notification_enqueue_status_total.labels.assert_called_once_with(
        notification_type="booking_assigned",
        channel="mission_updates",
        status="queued",
    )
    status_labels.inc.assert_called_once()


def test_metrics_03_prometheus_error_does_not_fail_enqueue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from services.events import fanout

    delay = MagicMock()
    monkeypatch.setattr(
        fanout, "should_send_night_notification", lambda *_a, **_k: True
    )
    monkeypatch.setattr(
        "services.notifications.push._check_duplicate_notification",
        lambda *_a, **_k: False,
    )
    monkeypatch.setattr(
        "services.monitoring.prometheus.inc_driver_push_channel",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        "tasks.notification_tasks.send_push_notification_task.delay",
        delay,
    )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise ValueError("Incorrect label names")

    monkeypatch.setattr(fanout, "track_notification_sent", _boom)

    ok = fanout._send_push_to_driver(
        20135,
        "Nouveau transport assigné",
        "Un transport vous a été assigné pour 16:00.",
        {"type": "booking_assigned", "booking_id": 40500},
    )

    assert ok is True
    delay.assert_called_once()
