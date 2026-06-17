"""Tests logs pipeline push chauffeur."""

from __future__ import annotations

import json
import logging

import pytest

from services.notifications.push_pipeline_log import log_driver_push_skipped


def test_log_driver_push_skipped_emits_structured_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    log_driver_push_skipped(
        reason="non_significant_change",
        driver_id=3,
        booking_id=35225,
        changes_keys=["internal_ref"],
    )
    assert any("[driver_push_pipeline]" in record.message for record in caplog.records)
    payload_line = next(
        record.message for record in caplog.records if "[driver_push_pipeline]" in record.message
    )
    payload = json.loads(payload_line.split("[driver_push_pipeline] ", 1)[1])
    assert payload["stage"] == "driver_push.skipped"
    assert payload["push_skipped_reason"] == "non_significant_change"
    assert payload["driver_id"] == 3
    assert payload["booking_id"] == 35225
    assert payload["changes_keys"] == ["internal_ref"]
