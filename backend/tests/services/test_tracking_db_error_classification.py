"""Tests matrice d'erreurs DB pour le consumer tracking ingest."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import (
    DataError,
    IntegrityError,
    InterfaceError,
    OperationalError,
    ProgrammingError,
)

from services.tracking.db_error_classification import (
    DbErrorAction,
    classify_db_error,
    is_infrastructure_db_error,
)
from services.tracking.ingest_consumer import (
    FatalTrackingConsumerError,
    TrackingIngestConsumer,
)


def test_operational_error_is_infrastructure():
    exc = OperationalError("stmt", {}, Exception("connection refused"))
    assert classify_db_error(exc) == DbErrorAction.INFRASTRUCTURE_RETRY
    assert is_infrastructure_db_error(exc) is True


def test_interface_error_is_infrastructure():
    exc = InterfaceError("stmt", {}, Exception("connection already closed"))
    assert classify_db_error(exc) == DbErrorAction.INFRASTRUCTURE_RETRY


def test_programming_error_is_fail_stop():
    exc = ProgrammingError("stmt", {}, Exception("relation does not exist"))
    assert classify_db_error(exc) == DbErrorAction.FAIL_STOP
    assert is_infrastructure_db_error(exc) is False


def test_integrity_unknown_is_fail_stop():
    exc = IntegrityError("stmt", {}, Exception("some obscure constraint xyz"))
    assert classify_db_error(exc) == DbErrorAction.FAIL_STOP


def test_integrity_location_event_id_unique_is_duplicate():
    exc = IntegrityError(
        "stmt",
        {},
        Exception(
            'duplicate key value violates unique constraint '
            'on (driver_id, location_event_id)'
        ),
    )
    assert classify_db_error(exc) == DbErrorAction.IDEMPOTENT_DUPLICATE


def test_integrity_check_constraint_is_dlq():
    exc = IntegrityError(
        "stmt",
        {},
        Exception("new row violates check constraint ck_lat_range"),
    )
    assert classify_db_error(exc) == DbErrorAction.DLQ


def test_data_error_message_attributable_is_dlq():
    exc = DataError(
        "stmt",
        {},
        Exception("value too long for type character varying(64)"),
    )
    assert classify_db_error(exc) == DbErrorAction.DLQ


def test_wrapped_cause_is_recognized():
    inner = OperationalError("stmt", {}, Exception("server closed the connection"))
    outer = RuntimeError("persist failed")
    outer.__cause__ = inner
    assert classify_db_error(outer) == DbErrorAction.INFRASTRUCTURE_RETRY


def _valid_record():
    return SimpleNamespace(
        topic="raw",
        partition=0,
        offset=42,
        timestamp=1_700_000_000_000,
        value={
            "driver_id": 7,
            "payload": {"latitude": 46.2, "longitude": 6.1},
            "source": "mobile",
        },
    )


def _consumer_for_persist(monkeypatch):
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_PERSIST_WITH_OUTBOX", True
    )
    monkeypatch.setattr(
        "services.tracking.ingest_consumer.TRACKING_INGEST_PERSIST_ENABLED", False
    )
    monkeypatch.setattr("services.tracking.ingest_consumer.KAFKA_MAX_RETRIES", 2)
    monkeypatch.setattr("services.tracking.ingest_consumer.KAFKA_RETRY_BACKOFF_MS", 1)
    consumer = TrackingIngestConsumer()
    consumer._commit_record = MagicMock()
    consumer._send_to_dlq_and_commit = MagicMock(return_value=True)
    consumer._observe_e2e_latency = MagicMock()
    return consumer


def test_process_operational_exhausted_fail_stop_no_dlq_no_commit(monkeypatch):
    consumer = _consumer_for_persist(monkeypatch)
    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise OperationalError("stmt", {}, Exception("connection refused"))

    monkeypatch.setattr(
        "services.tracking.persist_kafka_outbox.persist_driver_location_with_outbox_from_kafka",
        _boom,
    )
    with pytest.raises(FatalTrackingConsumerError) as ei:
        consumer._process_record(_valid_record())
    assert "db_infrastructure_exhausted" in str(ei.value)
    assert calls["n"] == 2
    consumer._commit_record.assert_not_called()
    consumer._send_to_dlq_and_commit.assert_not_called()


def test_process_programming_error_fail_stop_immediate(monkeypatch):
    consumer = _consumer_for_persist(monkeypatch)
    calls = {"n": 0}

    def _boom(*_a, **_k):
        calls["n"] += 1
        raise ProgrammingError("stmt", {}, Exception("relation missing"))

    monkeypatch.setattr(
        "services.tracking.persist_kafka_outbox.persist_driver_location_with_outbox_from_kafka",
        _boom,
    )
    with pytest.raises(FatalTrackingConsumerError):
        consumer._process_record(_valid_record())
    assert calls["n"] == 1
    consumer._commit_record.assert_not_called()
    consumer._send_to_dlq_and_commit.assert_not_called()


def test_process_unknown_integrity_fail_stop_no_commit(monkeypatch):
    consumer = _consumer_for_persist(monkeypatch)

    def _boom(*_a, **_k):
        raise IntegrityError("stmt", {}, Exception("obscure constraint abc"))

    monkeypatch.setattr(
        "services.tracking.persist_kafka_outbox.persist_driver_location_with_outbox_from_kafka",
        _boom,
    )
    with pytest.raises(FatalTrackingConsumerError):
        consumer._process_record(_valid_record())
    consumer._commit_record.assert_not_called()
    consumer._send_to_dlq_and_commit.assert_not_called()


def test_process_location_event_id_unique_commits_without_dlq(monkeypatch):
    consumer = _consumer_for_persist(monkeypatch)

    def _boom(*_a, **_k):
        raise IntegrityError(
            "stmt",
            {},
            Exception(
                "duplicate key value violates unique constraint "
                "on driver_location_events (driver_id, location_event_id)"
            ),
        )

    monkeypatch.setattr(
        "services.tracking.persist_kafka_outbox.persist_driver_location_with_outbox_from_kafka",
        _boom,
    )
    ok = consumer._process_record(_valid_record())
    assert ok is True
    consumer._commit_record.assert_called_once()
    consumer._send_to_dlq_and_commit.assert_not_called()
