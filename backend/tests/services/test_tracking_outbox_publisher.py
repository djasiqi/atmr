"""Outbox : ordre session_generation + sequence_id (gate Phase 1)."""

from __future__ import annotations

from unittest.mock import MagicMock

from services.tracking.outbox_publisher import TrackingOutboxPublisher


def test_publish_for_driver_orders_by_generation_then_sequence(monkeypatch):
    publisher = TrackingOutboxPublisher(engine=MagicMock())
    rows = [
        {
            "id": 1,
            "event_id": "e1",
            "location_event_id": "e1",
            "payload": {"sequence_id": 1001},
            "session_generation": 1,
            "sequence_id": 1001,
        },
        {
            "id": 2,
            "event_id": "e2",
            "location_event_id": "e2",
            "payload": {"sequence_id": 1002},
            "session_generation": 1,
            "sequence_id": 1002,
        },
    ]
    sent_keys: list[str] = []

    class _Conn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, statement, params=None):
            sql = str(statement)
            if "pg_try_advisory_lock" in sql:
                return MagicMock(scalar=lambda: True)
            if "pg_advisory_unlock" in sql:
                return MagicMock(scalar=lambda: True)
            if "FROM tracking_event_outbox" in sql and "SELECT id" in sql:
                result = MagicMock()
                result.mappings.return_value.all.return_value = rows
                return result
            return MagicMock()

        def commit(self):
            return None

    publisher._engine.connect = lambda: _Conn()  # type: ignore[method-assign]

    class _Future:
        def get(self, timeout=None):
            return True

    class _Producer:
        def send(self, topic, key=None, value=None):
            sent_keys.append(str(value.get("sequence_id")))
            return _Future()

    monkeypatch.setattr(publisher, "_ensure_producer", lambda: _Producer())
    count = publisher._publish_for_driver(7)
    assert count == 2
    assert sent_keys == ["1001", "1002"]


def test_compare_shadow_codes():
    from services.tracking.shadow_ingest import compare_shadow_vs_direct

    assert (
        compare_shadow_vs_direct(
            location_event_id="e1",
            shadow_payload=None,
            direct_payload={"latitude": 1},
        )
        == "shadow_missing_in_kafka"
    )
    assert (
        compare_shadow_vs_direct(
            location_event_id="e1",
            shadow_payload={"latitude": 1, "longitude": 2, "recorded_at": "t", "company_id": 1},
            direct_payload={"latitude": 1, "longitude": 2, "recorded_at": "t", "company_id": 1},
        )
        == "shadow_match"
    )
