from __future__ import annotations

from types import SimpleNamespace

from services.tracking.ingest_consumer import TrackingIngestConsumer


def _record():
    return SimpleNamespace(topic="raw", partition=0, offset=11, key="driver_1")


def test_dlq_exhaustion_force_commit_avoids_offset_stall(monkeypatch):
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
    monkeypatch.setattr(consumer, "_commit_current", lambda: commit_calls.append(1))
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
    monkeypatch.setattr(consumer, "_commit_current", lambda: None)
    ok = consumer._send_to_dlq_and_commit(
        record=_record(),
        key="driver_1",
        source_message={"driver_id": 1, "payload": {"latitude": 1, "longitude": 2}},
        error=RuntimeError("failed"),
        retry_count=3,
        error_type="transient_exhausted",
    )
    assert ok is False
    assert len(attempts) == 3
