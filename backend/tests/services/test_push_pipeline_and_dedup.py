"""Tests push pipeline logging and dedup."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


def test_log_driver_push_stage_emits_json(caplog):
    from services.notifications.push_pipeline_log import log_driver_push_stage

    with caplog.at_level("INFO"):
        log_driver_push_stage(
            "driver_push.publish",
            event_id="evt-1",
            booking_id=34071,
            driver_id=6855,
        )

    assert any("[driver_push_pipeline]" in r.message for r in caplog.records)
    payload = json.loads(
        caplog.records[-1].message.split("[driver_push_pipeline] ", 1)[1]
    )
    assert payload["stage"] == "driver_push.publish"
    assert payload["booking_id"] == 34071
    assert payload["driver_id"] == 6855


def test_claim_driver_booking_push_fail_open_without_redis():
    from services.notifications.push_driver_booking_dedup import (
        claim_driver_booking_push,
    )

    with patch(
        "services.notifications.push_driver_booking_dedup._redis", return_value=None
    ):
        assert claim_driver_booking_push(6855, 34071) is True


def test_claim_driver_booking_push_dedup_second_call():
    from services.notifications.push_driver_booking_dedup import (
        claim_driver_booking_push,
    )

    mock_redis = MagicMock()
    mock_redis.set.side_effect = [True, None]

    with patch(
        "services.notifications.push_driver_booking_dedup._redis",
        return_value=mock_redis,
    ):
        assert claim_driver_booking_push(6855, 34071) is True
        assert claim_driver_booking_push(6855, 34071) is False


def test_celery_health_ping_returns_ok():
    from tasks.health_tasks import celery_health_ping

    assert celery_health_ping.run() == "ok"
