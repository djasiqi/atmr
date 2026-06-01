"""Tests logs pipeline dedup/throttle (nom corrigé — ne teste pas la route silent-ack)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch


def test_should_skip_dedup_emits_pipeline_log(caplog):
    from services.notifications.dedup_throttle import should_skip_dedup

    mock_redis = MagicMock()
    mock_redis.set.return_value = None

    with patch("services.notifications.dedup_throttle._get_redis", return_value=mock_redis), patch(
        "services.notifications.metrics.record_dedup_hit"
    ):
        with caplog.at_level("INFO"):
            assert should_skip_dedup("driver", 42, "dedupe-key") is True

    assert any("[notification_pipeline]" in r.message for r in caplog.records)
    payload = json.loads(caplog.records[-1].message.split("[notification_pipeline] ", 1)[1])
    assert payload["event"] == "notification_deduped"
    assert payload["driver_id"] == 42


def test_should_skip_throttle_emits_pipeline_log(caplog):
    from services.notifications.dedup_throttle import should_skip_throttle

    mock_redis = MagicMock()
    mock_redis.eval.return_value = 4

    with patch("services.notifications.dedup_throttle._get_redis", return_value=mock_redis), patch(
        "services.notifications.metrics.record_throttle_block"
    ):
        with caplog.at_level("INFO"):
            assert should_skip_throttle("driver", 7, "scope", window_s=60, max_per_window=3) is True

    assert any("[notification_pipeline]" in r.message for r in caplog.records)
    payload = json.loads(caplog.records[-1].message.split("[notification_pipeline] ", 1)[1])
    assert payload["event"] == "notification_throttled"
    assert payload["driver_id"] == 7
