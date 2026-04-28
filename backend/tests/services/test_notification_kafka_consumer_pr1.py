"""Tests PR-1 : skip + commit, retries, DLQ (kafka_consumer notifications)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services.notifications import kafka_consumer as kc


def test_schema_skip_reason_push_missing_field():
    consumer = kc.KafkaConsumer.__new__(kc.KafkaConsumer)
    assert (
        consumer._schema_skip_reason(
            kc.KAFKA_TOPIC_NOTIFICATIONS,
            {"driver_id": 1, "title": "t"},
        )
        == "invalid_payload"
    )


def test_schema_skip_reason_push_ok():
    consumer = kc.KafkaConsumer.__new__(kc.KafkaConsumer)
    assert (
        consumer._schema_skip_reason(
            kc.KAFKA_TOPIC_NOTIFICATIONS,
            {"driver_id": 1, "title": "t", "body": "b"},
        )
        is None
    )


def test_is_transient_error():
    assert kc._is_transient_error(TimeoutError("timeout"))
    assert kc._is_transient_error(Exception("connection refused"))
    assert not kc._is_transient_error(ValueError("bad payload"))


def test_handle_record_skip_invalid_payload():
    consumer = kc.KafkaConsumer.__new__(kc.KafkaConsumer)
    consumer._consumer = MagicMock()
    record = MagicMock()
    record.topic = kc.KAFKA_TOPIC_NOTIFICATIONS
    record.value = {"driver_id": 1}
    record.partition = 0
    record.offset = 42

    with patch.object(consumer, "_log_notification_consumer_skip") as log_skip:
        consumer._handle_record(record)

    consumer._consumer.commit.assert_called_once()
    log_skip.assert_called_once()
    assert log_skip.call_args[0][1] == "invalid_payload"


def test_handle_record_dlq_on_persistent_failure():
    consumer = kc.KafkaConsumer.__new__(kc.KafkaConsumer)
    consumer._consumer = MagicMock()
    record = MagicMock()
    record.topic = kc.KAFKA_TOPIC_NOTIFICATIONS
    record.value = {"driver_id": 1, "title": "t", "body": "b", "data": {}}
    record.partition = 1
    record.offset = 99

    with (
        patch.object(
            consumer,
            "_process_message",
            side_effect=ValueError("not transient"),
        ),
        patch.object(consumer, "_send_to_dlq", return_value=True) as dlq,
    ):
        consumer._handle_record(record)

    dlq.assert_called_once()
    consumer._consumer.commit.assert_called_once()


def test_handle_record_no_commit_when_dlq_fails():
    consumer = kc.KafkaConsumer.__new__(kc.KafkaConsumer)
    consumer._consumer = MagicMock()
    record = MagicMock()
    record.topic = kc.KAFKA_TOPIC_NOTIFICATIONS
    record.value = {"driver_id": 1, "title": "t", "body": "b", "data": {}}
    record.partition = 0
    record.offset = 1

    with (
        patch.object(
            consumer,
            "_process_message",
            side_effect=ValueError("not transient"),
        ),
        patch.object(consumer, "_send_to_dlq", return_value=False),
    ):
        consumer._handle_record(record)

    consumer._consumer.commit.assert_not_called()
