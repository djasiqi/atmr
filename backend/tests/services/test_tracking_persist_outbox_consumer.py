"""Phase 1 : consumer RAW commit après outbox, sans publish processed."""

from __future__ import annotations

from types import SimpleNamespace

from services.tracking.ingest_consumer import TrackingIngestConsumer


def _record(value: dict):
    return SimpleNamespace(
        topic="raw",
        partition=0,
        offset=42,
        key="driver_1",
        timestamp=1_700_000_000_000,
        value=value,
    )


def test_outbox_path_commits_without_processed_publish(monkeypatch):
    consumer = TrackingIngestConsumer()
    publish_calls: list[str] = []
    commit_calls: list[int] = []

    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_PERSIST_WITH_OUTBOX", True
    )
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_INGEST_PERSIST_ENABLED", False
    )

    def _persist(message_obj, *, driver_id: int):
        return (
            {**message_obj, "location_event_id": "e1"},
            {"status": "persisted", "location_event_id": "e1"},
        )

    monkeypatch.setattr(
        "services.tracking.persist_kafka_outbox.persist_driver_location_with_outbox_from_kafka",
        _persist,
    )
    monkeypatch.setattr(
        consumer,
        "_publish_with_ack",
        lambda **kwargs: publish_calls.append(kwargs["topic"]),
    )
    monkeypatch.setattr(
        consumer, "_commit_record", lambda record: commit_calls.append(record.offset)
    )
    monkeypatch.setattr(consumer, "_observe_e2e_latency", lambda _m: None)
    monkeypatch.setattr(consumer, "_is_valid", lambda _m: True)

    ok = consumer._process_record(
        _record(
            {
                "driver_id": 1,
                "company_id": 9,
                "source": "http_batch",
                "payload": {
                    "latitude": 46.2,
                    "longitude": 6.1,
                    "tracking_session_id": "s1",
                    "sequence_id": 1,
                    "location_event_id": "e1",
                },
            }
        )
    )
    assert ok is True
    assert commit_calls == [42]
    assert publish_calls == []


def test_session_mismatch_goes_to_dlq(monkeypatch):
    from services.tracking.persist_kafka_outbox import PersistKafkaOutboxError

    consumer = TrackingIngestConsumer()
    dlq_calls: list[str] = []
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_PERSIST_WITH_OUTBOX", True
    )
    monkeypatch.setattr(consumer, "_is_valid", lambda _m: True)

    def _boom(*_a, **_k):
        raise PersistKafkaOutboxError("session_generation_mismatch")

    monkeypatch.setattr(
        "services.tracking.persist_kafka_outbox.persist_driver_location_with_outbox_from_kafka",
        _boom,
    )
    monkeypatch.setattr(
        consumer,
        "_send_to_dlq_and_commit",
        lambda **kwargs: dlq_calls.append(kwargs["error_type"]) or True,
    )

    ok = consumer._process_record(
        _record(
            {
                "driver_id": 1,
                "payload": {"latitude": 1.0, "longitude": 2.0},
            }
        )
    )
    assert ok is True
    assert dlq_calls == ["session_generation_mismatch"]
