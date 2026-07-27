from __future__ import annotations

import pytest
from types import SimpleNamespace

from services.tracking.ingest_consumer import (
    FatalTrackingConsumerError,
    TrackingIngestConsumer,
)


def _record():
    return SimpleNamespace(topic="raw", partition=0, offset=11, key="driver_1")


def test_dlq_exhaustion_raises_fatal_without_commit(monkeypatch):
    """Phase 0B : force-commit désactivé → FatalTrackingConsumerError, pas de commit."""
    consumer = TrackingIngestConsumer()
    commit_calls: list[int] = []
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_DLQ_PUBLISH_MAX_ATTEMPTS", 2
    )
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE", False
    )
    monkeypatch.setattr(
        consumer,
        "_publish_with_ack",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("dlq down")),
    )
    monkeypatch.setattr(
        consumer, "_commit_record", lambda record: commit_calls.append(1)
    )
    with pytest.raises(FatalTrackingConsumerError) as exc_info:
        consumer._send_to_dlq_and_commit(
            record=_record(),
            key="driver_1",
            source_message={"driver_id": 1, "payload": {"latitude": 1, "longitude": 2}},
            error=RuntimeError("failed"),
            retry_count=3,
            error_type="transient_exhausted",
        )
    assert exc_info.value.partition == 0
    assert exc_info.value.offset == 11
    assert len(commit_calls) == 0


def test_dlq_retries_are_bounded(monkeypatch):
    consumer = TrackingIngestConsumer()
    attempts: list[int] = []
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_DLQ_PUBLISH_MAX_ATTEMPTS", 3
    )
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE", False
    )

    def _boom(**kwargs):
        attempts.append(1)
        raise RuntimeError("still down")

    monkeypatch.setattr(consumer, "_publish_with_ack", _boom)
    monkeypatch.setattr(consumer, "_commit_record", lambda record: None)
    with pytest.raises(FatalTrackingConsumerError):
        consumer._send_to_dlq_and_commit(
            record=_record(),
            key="driver_1",
            source_message={"driver_id": 1, "payload": {"latitude": 1, "longitude": 2}},
            error=RuntimeError("failed"),
            retry_count=3,
            error_type="transient_exhausted",
        )
    assert len(attempts) == 3


def test_legacy_force_commit_still_works_when_explicitly_enabled(monkeypatch):
    """Chaos / rollback uniquement — flag true autorise encore le force-commit."""
    consumer = TrackingIngestConsumer()
    commit_calls: list[int] = []
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_DLQ_PUBLISH_MAX_ATTEMPTS", 2
    )
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_DLQ_FORCE_COMMIT_ON_FAILURE", True
    )
    monkeypatch.setattr(
        consumer,
        "_publish_with_ack",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("dlq down")),
    )
    monkeypatch.setattr(
        consumer, "_commit_record", lambda record: commit_calls.append(1)
    )
    ok = consumer._send_to_dlq_and_commit(
        record=_record(),
        key="driver_1",
        source_message={"driver_id": 1, "payload": {"latitude": 1, "longitude": 2}},
        error=RuntimeError("failed"),
        retry_count=3,
        error_type="transient_exhausted",
    )
    assert ok is True
    assert len(commit_calls) == 1
