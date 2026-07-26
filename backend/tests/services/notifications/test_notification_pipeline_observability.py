"""Tests observabilité pipeline notifications."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from services.notifications.notification_pipeline_observability import (
    build_idempotency_key,
    claim_idempotency_key,
    log_notification_pipeline_event,
)


def test_log_notification_pipeline_event_emits_json(caplog):
    with caplog.at_level("INFO"):
        log_notification_pipeline_event(
            "notification_created",
            notification_id="n-1",
            booking_id=42,
            driver_id=7,
            notification_type="new_booking",
            correlation_id="corr-1",
        )

    record = caplog.records[-1]
    payload = json.loads(record.message.split("[notification_pipeline] ", 1)[1])
    assert payload["event"] == "notification_created"
    assert payload["pipeline_stage"] == "fanout"
    assert payload["booking_id"] == 42
    assert payload["driver_id"] == 7


def test_build_idempotency_key_stable():
    k1 = build_idempotency_key(
        driver_id=1,
        notification_type="booking",
        title="T",
        body="B",
        data={"booking_id": 99},
    )
    k2 = build_idempotency_key(
        driver_id=1,
        notification_type="booking",
        title="T",
        body="B",
        data={"booking_id": 99},
    )
    assert k1 == k2
    assert k1.startswith("push:idempotency:")


def test_claim_idempotency_key_second_call_false():
    mock_redis = MagicMock()
    mock_redis.set.side_effect = [True, None]

    with patch("ext.redis_client", mock_redis):
        assert claim_idempotency_key("push:idempotency:abc") is True
        assert claim_idempotency_key("push:idempotency:abc") is False


def test_log_notification_mobile_received_records_delivered_metric(caplog):
    with (
        patch(
            "services.notifications.notification_pipeline_observability._record_business_metric"
        ) as mock_metric,
        caplog.at_level("INFO"),
    ):
        log_notification_pipeline_event(
            "notification_mobile_received",
            notification_id="n-99",
            booking_id=12,
            driver_id=3,
            notification_type="booking",
            correlation_id="corr-mobile",
        )

    mock_metric.assert_called_once_with(
        "notification_mobile_received", notification_type="booking"
    )
