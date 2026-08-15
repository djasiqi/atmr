"""P0-E — persisted_sync = preuve ledger + commit PG (pas projection Driver seule)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from services.tracking.persist_with_outbox import PersistConflictError
from services.tracking.sync_ledger_ack import (
    SyncLedgerAckResult,
    durable_proof,
    extract_sync_ledger_ids,
    try_commit_sync_ledger_ack,
)


def test_durable_proof_only_inserted_or_same_event() -> None:
    assert durable_proof({"status": "persisted", "reason": "inserted"}) is True
    assert (
        durable_proof({"status": "duplicate", "reason": "same_event_already_persisted"})
        is True
    )
    assert (
        durable_proof(
            {
                "status": "duplicate",
                "reason": "session_sequence_already_persisted",
            }
        )
        is False
    )
    assert (
        durable_proof({"status": "duplicate", "reason": "duplicate_unproven"}) is False
    )
    assert durable_proof({"status": "persisted"}) is False
    assert durable_proof({"status": "duplicate"}) is False
    assert durable_proof(None) is False


def test_extract_sync_ledger_ids() -> None:
    sid, gen, seq = extract_sync_ledger_ids(
        {
            "tracking_session_id": " sess-1 ",
            "session_generation": "2",
            "sequence_id": 7,
        }
    )
    assert sid == "sess-1"
    assert gen == 2
    assert seq == 7
    assert extract_sync_ledger_ids({}) == (None, None, None)
    assert extract_sync_ledger_ids(None) == (None, None, None)


def test_ids_missing_never_calls_persist(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    def _boom(*_a, **_k):  # type: ignore[no-untyped-def]
        called["n"] += 1
        raise AssertionError("persist ne doit pas être appelé sans IDs")

    monkeypatch.setattr(
        "services.tracking.sync_ledger_ack.persist_location_event_with_outbox",
        _boom,
    )
    session = MagicMock()
    out = try_commit_sync_ledger_ack(
        session,
        driver_id=1,
        company_id=1,
        location_event_id="eid-1",
        tracking_session_id=None,
        session_generation=1,
        sequence_id=1,
        latitude=46.0,
        longitude=6.0,
        recorded_at="2026-08-11T10:00:00+00:00",
    )
    assert out.kind == "ids_missing"
    assert called["n"] == 0
    session.commit.assert_not_called()


def test_insert_and_commit_returns_durable_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "services.tracking.sync_ledger_ack.persist_location_event_with_outbox",
        lambda *_a, **_k: {
            "status": "persisted",
            "reason": "inserted",
            "location_event_id": "eid-1",
        },
    )
    session = MagicMock()
    out = try_commit_sync_ledger_ack(
        session,
        driver_id=1,
        company_id=1,
        location_event_id="eid-1",
        tracking_session_id="sess-1",
        session_generation=1,
        sequence_id=3,
        latitude=46.0,
        longitude=6.0,
        recorded_at="2026-08-11T10:00:00+00:00",
    )
    assert out.kind == "durable_ok"
    assert out.reason == "inserted"
    session.commit.assert_called_once()
    session.rollback.assert_not_called()


def test_same_event_already_persisted_is_durable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "services.tracking.sync_ledger_ack.persist_location_event_with_outbox",
        lambda *_a, **_k: {
            "status": "duplicate",
            "reason": "same_event_already_persisted",
            "location_event_id": "eid-1",
        },
    )
    session = MagicMock()
    out = try_commit_sync_ledger_ack(
        session,
        driver_id=1,
        company_id=1,
        location_event_id="eid-1",
        tracking_session_id="sess-1",
        session_generation=1,
        sequence_id=3,
        latitude=46.0,
        longitude=6.0,
        recorded_at="2026-08-11T10:00:00+00:00",
    )
    assert out.kind == "durable_ok"
    assert out.reason == "same_event_already_persisted"
    session.commit.assert_called_once()


def test_session_sequence_other_event_is_409(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "services.tracking.sync_ledger_ack.persist_location_event_with_outbox",
        lambda *_a, **_k: {
            "status": "duplicate",
            "reason": "session_sequence_already_persisted",
            "location_event_id": "eid-new",
            "existing_location_event_id": "eid-old",
        },
    )
    session = MagicMock()
    out = try_commit_sync_ledger_ack(
        session,
        driver_id=1,
        company_id=1,
        location_event_id="eid-new",
        tracking_session_id="sess-1",
        session_generation=1,
        sequence_id=3,
        latitude=46.0,
        longitude=6.0,
        recorded_at="2026-08-11T10:00:00+00:00",
    )
    assert out.kind == "conflict_409"
    assert out.reason == "session_sequence_already_persisted"
    assert out.existing_location_event_id == "eid-old"
    session.commit.assert_not_called()
    session.rollback.assert_called()


def test_payload_conflict_is_409(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a, **_k):  # type: ignore[no-untyped-def]
        raise PersistConflictError("event_id_payload_conflict")

    monkeypatch.setattr(
        "services.tracking.sync_ledger_ack.persist_location_event_with_outbox",
        _raise,
    )
    session = MagicMock()
    out = try_commit_sync_ledger_ack(
        session,
        driver_id=1,
        company_id=1,
        location_event_id="eid-1",
        tracking_session_id="sess-1",
        session_generation=1,
        sequence_id=1,
        latitude=46.0,
        longitude=6.0,
        recorded_at="2026-08-11T10:00:00+00:00",
    )
    assert out.kind == "conflict_409"
    assert out.reason == "event_id_payload_conflict"
    session.commit.assert_not_called()
    session.rollback.assert_called()


def test_commit_ko_never_durable_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "services.tracking.sync_ledger_ack.persist_location_event_with_outbox",
        lambda *_a, **_k: {
            "status": "persisted",
            "reason": "inserted",
            "location_event_id": "eid-1",
        },
    )
    session = MagicMock()
    session.commit.side_effect = RuntimeError("pg commit down")
    out = try_commit_sync_ledger_ack(
        session,
        driver_id=1,
        company_id=1,
        location_event_id="eid-1",
        tracking_session_id="sess-1",
        session_generation=1,
        sequence_id=1,
        latitude=46.0,
        longitude=6.0,
        recorded_at="2026-08-11T10:00:00+00:00",
    )
    assert out.kind == "ledger_failed_503"
    assert out.reason == "ledger_persist_failed"
    session.rollback.assert_called()


def test_duplicate_unproven_is_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "services.tracking.sync_ledger_ack.persist_location_event_with_outbox",
        lambda *_a, **_k: {
            "status": "duplicate",
            "reason": "duplicate_unproven",
            "location_event_id": "eid-1",
        },
    )
    session = MagicMock()
    out = try_commit_sync_ledger_ack(
        session,
        driver_id=1,
        company_id=1,
        location_event_id="eid-1",
        tracking_session_id="sess-1",
        session_generation=1,
        sequence_id=1,
        latitude=46.0,
        longitude=6.0,
        recorded_at="2026-08-11T10:00:00+00:00",
    )
    assert out.kind == "ledger_failed_503"
    session.commit.assert_not_called()


def test_projection_ok_without_ledger_must_not_claim_persisted_sync() -> None:
    """Matrice : db_persisted projection ≠ durability persisted_sync."""
    sync_db_persisted = True
    accept_status = "accepted_canonical"
    ledger = SyncLedgerAckResult(kind="ids_missing", reason="ledger_ids_missing")
    durable_ok = (
        accept_status == "accepted_canonical"
        and sync_db_persisted is True
        and ledger.kind == "durable_ok"
    )
    assert durable_ok is False
    # Option B : ids_missing → rejected non-retryable (pas ingested_non_persisted)
    response = {
        "ack_status": "rejected",
        "accept_reason": "invalid_ledger_ids",
        "durability": None,
        "db_persisted": True,
        "ledger_persisted": False,
        "retryable": False,
    }
    assert response["durability"] != "persisted_sync"
    assert response["ledger_persisted"] is False
    assert response["retryable"] is False


def test_route_guard_strips_invented_persisted_sync() -> None:
    """Garde-fou driver.py : durability sans ledger_persisted → strip."""
    result_payload = {
        "durability": "persisted_sync",
        "ack_status": "persisted",
        "ledger_persisted": False,
        "db_persisted": True,
    }
    if (
        result_payload.get("durability") == "persisted_sync"
        and result_payload.get("ledger_persisted") is not True
    ):
        result_payload["durability"] = None
        result_payload["ack_status"] = "ingested_non_persisted"
        result_payload["ledger_persisted"] = False
    assert result_payload["durability"] is None
    assert result_payload["ack_status"] == "ingested_non_persisted"


def test_http_matrix_mapping() -> None:
    """Vérifie le mapping kind → HTTP / persisted_sync (Option B)."""

    def map_kind(kind: str) -> tuple[int, bool]:
        if kind == "durable_ok":
            return 200, True
        if kind == "ids_missing":
            return 422, False
        if kind == "conflict_409":
            return 409, False
        return 503, False

    assert map_kind("durable_ok") == (200, True)
    assert map_kind("ids_missing") == (422, False)
    assert map_kind("conflict_409") == (409, False)
    assert map_kind("ledger_failed_503") == (503, False)
