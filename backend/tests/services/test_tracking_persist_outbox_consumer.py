"""Phase 1 : consumer RAW commit après outbox, sans publish processed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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


def test_sequence_duplicate_skip_commits_offset(monkeypatch):
    """Conflit session/sequence traité en duplicate → offset avance (dépoison)."""
    consumer = TrackingIngestConsumer()
    commit_calls: list[int] = []
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_PERSIST_WITH_OUTBOX", True
    )
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_INGEST_PERSIST_ENABLED", False
    )

    def _persist(message_obj, *, driver_id: int):
        return (
            {**message_obj, "location_event_id": "e-new"},
            {
                "status": "duplicate",
                "location_event_id": "e-new",
                "reason": "session_sequence_already_persisted",
            },
        )

    monkeypatch.setattr(
        "services.tracking.persist_kafka_outbox.persist_driver_location_with_outbox_from_kafka",
        _persist,
    )
    monkeypatch.setattr(
        consumer, "_commit_record", lambda record: commit_calls.append(record.offset)
    )
    monkeypatch.setattr(consumer, "_observe_e2e_latency", lambda _m: None)
    monkeypatch.setattr(consumer, "_is_valid", lambda _m: True)

    ok = consumer._process_record(
        _record(
            {
                "driver_id": 3,
                "company_id": 1,
                "source": "http",
                "payload": {
                    "latitude": 46.1,
                    "longitude": 6.1,
                    "tracking_session_id": "http-legacy-3",
                    "sequence_id": 3,
                    "location_event_id": "e-new",
                },
            }
        )
    )
    assert ok is True
    assert commit_calls == [42]


@pytest.mark.parametrize("mode", ["enforce_mission", "strict"])
def test_firewall_exception_fail_closed_zero_pg_outbox(monkeypatch, mode: str):
    """Exception envelope/admit en enforce/strict → 0 PG, 0 outbox, 0 persist-success."""
    consumer = TrackingIngestConsumer()
    persist_calls: list[str] = []
    persist_success_calls: list[str] = []
    commit_calls: list[int] = []

    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_PERSIST_WITH_OUTBOX", True
    )
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_INGEST_PERSIST_ENABLED", False
    )
    monkeypatch.setattr(
        "services.tracking.mission_tracking_firewall.get_mission_firewall_mode",
        lambda: mode,
    )

    def _admit_boom(*_a, **_k):
        raise RuntimeError("firewall_simulated_failure")

    monkeypatch.setattr(
        "services.tracking.admission_gate.admit_mission_live_payload",
        _admit_boom,
    )
    monkeypatch.setattr(
        "services.tracking.persist_kafka_outbox.persist_driver_location_with_outbox_from_kafka",
        lambda *_a, **_k: persist_calls.append("pg_outbox") or ({}, {}),
    )
    monkeypatch.setattr(
        "services.tracking.async_circuit.mark_consumer_persist_success",
        lambda: persist_success_calls.append("success"),
    )
    monkeypatch.setattr(
        consumer, "_commit_record", lambda record: commit_calls.append(record.offset)
    )
    monkeypatch.setattr(consumer, "_observe_e2e_latency", lambda _m: None)
    monkeypatch.setattr(consumer, "_is_valid", lambda _m: True)

    ok = consumer._process_record(
        _record(
            {
                "driver_id": 7,
                "company_id": 1,
                "source": "http",
                "payload": {
                    "latitude": 46.2,
                    "longitude": 6.1,
                    "recorded_at": "2026-08-12T12:00:00Z",
                    "location_event_id": "evt-x",
                    "mission_id": 100,
                    "location_mode": "mission_live",
                },
                "ingress_contract": {
                    "recorded_at_present": True,
                    "location_event_id_present": True,
                    "mission_id_present": True,
                    "tracking_session_id_present": False,
                    "session_generation_present": False,
                    "sequence_id_present": False,
                },
            }
        )
    )

    assert ok is True
    assert persist_calls == [], "aucune écriture PG/outbox après exception firewall"
    assert persist_success_calls == [], "aucun mark_consumer_persist_success"
    # Offset avance (dépoison) sans passer par le chemin persist-success
    assert commit_calls == [42]


def test_firewall_exception_fail_open_when_firewall_off(monkeypatch):
    """Mode off : exception admit ne doit pas bloquer le persist (fail-open)."""
    consumer = TrackingIngestConsumer()
    persist_calls: list[str] = []
    commit_calls: list[int] = []

    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_PERSIST_WITH_OUTBOX", True
    )
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_INGEST_PERSIST_ENABLED", False
    )
    monkeypatch.setattr(
        "services.tracking.mission_tracking_firewall.get_mission_firewall_mode",
        lambda: "off",
    )
    monkeypatch.setattr(
        "services.tracking.admission_gate.admit_mission_live_payload",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    def _persist(message_obj, *, driver_id: int):
        persist_calls.append("pg_outbox")
        return (
            {**message_obj, "location_event_id": "e1"},
            {"status": "persisted", "location_event_id": "e1"},
        )

    monkeypatch.setattr(
        "services.tracking.persist_kafka_outbox.persist_driver_location_with_outbox_from_kafka",
        _persist,
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
                "source": "http",
                "payload": {"latitude": 46.2, "longitude": 6.1},
            }
        )
    )
    assert ok is True
    assert persist_calls == ["pg_outbox"]
    assert commit_calls == [42]
