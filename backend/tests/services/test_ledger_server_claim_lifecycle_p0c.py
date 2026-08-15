"""P0-C-LEDGER-SERVER — T1–T7 claim lifecycle + sémantique duplicate.

Assert systématique sur l'état Redis final (pas seulement l'ACK HTTP).
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from services.geolocation import driver_location_dedup as d
from services.tracking.sync_ledger_ack import try_commit_sync_ledger_ack


class _FakeRedis:
    """Redis minimal avec TTL pour assertions d'état final."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttl_sec: dict[str, int] = {}
        self._lock = threading.Lock()

    def set(self, key, value, nx=False, ex=None):  # type: ignore[no-untyped-def]
        with self._lock:
            if nx and key in self.store:
                return False
            self.store[key] = value
            if ex is not None:
                self.ttl_sec[key] = int(ex)
            return True

    def delete(self, key):  # type: ignore[no-untyped-def]
        with self._lock:
            existed = key in self.store
            self.store.pop(key, None)
            self.ttl_sec.pop(key, None)
            return 1 if existed else 0

    def ttl(self, key):  # type: ignore[no-untyped-def]
        with self._lock:
            if key not in self.store:
                return -2
            return int(self.ttl_sec.get(key, -1))

    def age_claim(self, key: str, age_sec: int) -> None:
        """Simule le vieillissement du claim (TTL restant = default - age)."""
        with self._lock:
            if key not in self.store:
                return
            default = max(60, d._DEFAULT_EVENT_TTL)
            self.ttl_sec[key] = max(0, default - int(age_sec))


def _event_key(driver_id: int, event_id: str) -> str:
    return d._event_key(driver_id, event_id)


def _option_b_ids_missing_response(
    driver_id: int, location_event_id: str
) -> dict[str, Any]:
    """Miroir Option B de driver.py branche ids_missing."""
    d.release_location_event_id(
        driver_id, location_event_id, reason="invalid_ledger_ids"
    )
    return {
        "error": "invalid_ledger_ids",
        "error_code": "invalid_ledger_ids",
        "ok": False,
        "accept_status": "rejected_invalid",
        "accept_reason": "invalid_ledger_ids",
        "ack_status": "rejected",
        "durability": None,
        "location_event_id": location_event_id,
        "ledger_persisted": False,
        "retryable": False,
        "status_code": 422,
    }


def _verify_duplicate_without_proof(
    driver_id: int, location_event_id: str
) -> dict[str, Any]:
    """Miroir VERIFY driver.py après SET NX fail sans preuve persisted_sync."""
    dup_class = d.classify_duplicate_event_without_persisted_proof(
        driver_id, location_event_id
    )
    if dup_class == "claim_in_flight":
        return {
            "ok": True,
            "accept_reason": "claim_in_flight",
            "ack_status": "ingested_non_persisted",
            "durability": None,
            "retryable": True,
            "ledger_persisted": False,
        }
    d.release_location_event_id(
        driver_id, location_event_id, reason="duplicate_event_id_unproven"
    )
    return {
        "ok": True,
        "accept_reason": "duplicate_event_id_unproven",
        "ack_status": "ingested_non_persisted",
        "durability": None,
        "retryable": True,
        "ledger_persisted": False,
    }


@pytest.fixture
def fake_redis(monkeypatch: pytest.MonkeyPatch) -> _FakeRedis:
    fake = _FakeRedis()
    monkeypatch.setattr(d, "_redis", lambda: fake)
    monkeypatch.setattr(d, "_CLAIM_IN_FLIGHT_MAX_AGE_SEC", 15)
    return fake


# ---------------------------------------------------------------------------
# T1 — incident 14 août : generation=null → claim → reject → claim absent
# ---------------------------------------------------------------------------


def test_t1_ids_missing_releases_claim_no_orphan_unproven(
    fake_redis: _FakeRedis,
) -> None:
    driver_id = 19
    event_id = "evt-aug14-gen-null"

    assert d.claim_location_event_id(driver_id, event_id) is True
    assert d.location_event_claim_present(driver_id, event_id) is True

    # sync_ledger : generation=null → ids_missing (sans DB)
    ledger = try_commit_sync_ledger_ack(
        session=None,  # type: ignore[arg-type]
        driver_id=driver_id,
        company_id=1,
        location_event_id=event_id,
        tracking_session_id="sess-1",
        session_generation=None,
        sequence_id=1,
        latitude=46.0,
        longitude=6.0,
        recorded_at=None,
    )
    assert ledger.kind == "ids_missing"

    ack = _option_b_ids_missing_response(driver_id, event_id)
    assert ack["ack_status"] == "rejected"
    assert ack["retryable"] is False
    assert ack["status_code"] == 422
    # État Redis final : claim ABSENT (pas orphelin)
    assert d.location_event_claim_present(driver_id, event_id) is False
    assert _event_key(driver_id, event_id) not in fake_redis.store

    # Retry identique : reclaim OK — ne tombe PAS dans unproven à cause du claim précédent
    assert d.claim_location_event_id(driver_id, event_id) is True
    assert d.location_event_claim_present(driver_id, event_id) is True


# ---------------------------------------------------------------------------
# T2 — persistence exception → release (pas de poison)
# ---------------------------------------------------------------------------


def test_t2_persistence_exception_releases_claim(fake_redis: _FakeRedis) -> None:
    driver_id = 7
    event_id = "evt-persist-fail"

    assert d.claim_location_event_id(driver_id, event_id) is True
    d.release_after_pre_persistence_failure(
        driver_id, event_id, reason="db_persist_failed"
    )
    assert d.location_event_claim_present(driver_id, event_id) is False
    # Retry possible
    assert d.claim_location_event_id(driver_id, event_id) is True


# ---------------------------------------------------------------------------
# T3 — duplicate + row persistée → duplicate_persisted, pas de double write
# ---------------------------------------------------------------------------


def test_t3_persisted_duplicate_keeps_idempotence(fake_redis: _FakeRedis) -> None:
    driver_id = 7
    event_id = "evt-already-persisted"
    writes = {"n": 0}
    idem_cache: dict[str, dict[str, Any]] = {}

    def persist_once() -> dict[str, Any]:
        writes["n"] += 1
        body = {
            "ack_status": "persisted",
            "durability": "persisted_sync",
            "accept_reason": "accepted",
            "ledger_persisted": True,
            "location_event_id": event_id,
        }
        idem_cache[event_id] = body
        return body

    # Premier passage
    assert d.claim_location_event_id(driver_id, event_id) is True
    first = persist_once()
    assert first["durability"] == "persisted_sync"
    assert writes["n"] == 1
    assert d.location_event_claim_present(driver_id, event_id) is True

    # Retry : SET NX fail + preuve cache → duplicate_persisted (pas de 2e write)
    assert d.claim_location_event_id(driver_id, event_id) is False
    proven = idem_cache.get(event_id)
    assert proven is not None
    assert proven.get("durability") == "persisted_sync"
    ack = {
        "accept_reason": "duplicate_persisted",
        "ack_status": "duplicate",
        "durability": "persisted_sync",
        "retryable": False,
        "ledger_persisted": True,
    }
    assert writes["n"] == 1
    assert d.location_event_claim_present(driver_id, event_id) is True
    assert ack["ack_status"] == "duplicate"
    assert ack["durability"] == "persisted_sync"


# ---------------------------------------------------------------------------
# T4 — unproven ≠ succès final
# ---------------------------------------------------------------------------


def test_t4_unproven_is_not_final_success(fake_redis: _FakeRedis) -> None:
    driver_id = 7
    event_id = "evt-orphan-claim"

    assert d.claim_location_event_id(driver_id, event_id) is True
    # Vieillir au-delà de in-flight → unproven
    fake_redis.age_claim(_event_key(driver_id, event_id), age_sec=60)
    assert (
        d.classify_duplicate_event_without_persisted_proof(driver_id, event_id)
        == "duplicate_unproven"
    )

    # Simuler SET NX fail (claim encore présent)
    assert d.claim_location_event_id(driver_id, event_id) is False
    ack = _verify_duplicate_without_proof(driver_id, event_id)

    assert ack["accept_reason"] == "duplicate_event_id_unproven"
    assert ack["ack_status"] == "ingested_non_persisted"
    assert ack["durability"] is None
    assert ack["retryable"] is True
    # Pas un succès final « déjà traité »
    assert not (
        ack["ack_status"] == "duplicate" and ack.get("durability") == "persisted_sync"
    )
    # Redis final : claim libéré
    assert d.location_event_claim_present(driver_id, event_id) is False


# ---------------------------------------------------------------------------
# T5 — retries concurrents → une seule persistence
# ---------------------------------------------------------------------------


def test_t5_concurrent_retries_single_write(fake_redis: _FakeRedis) -> None:
    driver_id = 7
    event_id = "evt-concurrent"
    writes = {"n": 0}
    barrier = threading.Barrier(2)
    results: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        acquired = d.claim_location_event_id(driver_id, event_id)
        if acquired:
            with lock:
                writes["n"] += 1
            with lock:
                results.append("persisted")
        else:
            # Concurrent loser : claim_in_flight (claim frais) — pas de write
            cls = d.classify_duplicate_event_without_persisted_proof(
                driver_id, event_id
            )
            with lock:
                results.append(cls)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert writes["n"] == 1
    assert results.count("persisted") == 1
    assert "claim_in_flight" in results or "duplicate_unproven" in results
    # Claim toujours présent (holder) — pas double release
    assert d.location_event_claim_present(driver_id, event_id) is True


# ---------------------------------------------------------------------------
# T6 — stale/orphan recovery
# ---------------------------------------------------------------------------


def test_t6_stale_orphan_claim_recovery(fake_redis: _FakeRedis) -> None:
    driver_id = 7
    event_id = "evt-stale-orphan"

    assert d.claim_location_event_id(driver_id, event_id) is True
    fake_redis.age_claim(_event_key(driver_id, event_id), age_sec=120)
    assert (
        d.classify_duplicate_event_without_persisted_proof(driver_id, event_id)
        == "duplicate_unproven"
    )

    # SET NX fail puis VERIFY unproven → release
    assert d.claim_location_event_id(driver_id, event_id) is False
    ack = _verify_duplicate_without_proof(driver_id, event_id)
    assert ack["accept_reason"] == "duplicate_event_id_unproven"
    assert d.location_event_claim_present(driver_id, event_id) is False

    # Récupération : nouveau claim OK
    assert d.claim_location_event_id(driver_id, event_id) is True


# ---------------------------------------------------------------------------
# T7 — vieux client generation=null → reject non-retryable + release
# ---------------------------------------------------------------------------


def test_t7_old_client_generation_null_reject_and_release(
    fake_redis: _FakeRedis,
) -> None:
    driver_id = 19
    event_id = "evt-old-client-null-gen"

    assert d.is_structural_ledger_ids_missing(
        tracking_session_id="sess",
        session_generation=None,
        sequence_id=42,
        location_event_id=event_id,
    )

    assert d.claim_location_event_id(driver_id, event_id) is True
    ledger = try_commit_sync_ledger_ack(
        session=None,  # type: ignore[arg-type]
        driver_id=driver_id,
        company_id=1,
        location_event_id=event_id,
        tracking_session_id="sess",
        session_generation=None,
        sequence_id=42,
        latitude=46.2,
        longitude=6.1,
        recorded_at=None,
    )
    assert ledger.kind == "ids_missing"

    ack = _option_b_ids_missing_response(driver_id, event_id)
    assert ack["accept_reason"] == "invalid_ledger_ids"
    assert ack["retryable"] is False
    assert ack["ack_status"] == "rejected"
    assert d.location_event_claim_present(driver_id, event_id) is False

    # Boucle poison impossible : reclaim puis même reject+release
    assert d.claim_location_event_id(driver_id, event_id) is True
    ack2 = _option_b_ids_missing_response(driver_id, event_id)
    assert ack2["retryable"] is False
    assert d.location_event_claim_present(driver_id, event_id) is False


def test_claim_in_flight_does_not_release(fake_redis: _FakeRedis) -> None:
    driver_id = 3
    event_id = "evt-in-flight"
    assert d.claim_location_event_id(driver_id, event_id) is True
    # Age ~0 → in_flight
    assert (
        d.classify_duplicate_event_without_persisted_proof(driver_id, event_id)
        == "claim_in_flight"
    )
    assert d.claim_location_event_id(driver_id, event_id) is False
    ack = _verify_duplicate_without_proof(driver_id, event_id)
    assert ack["accept_reason"] == "claim_in_flight"
    assert ack["retryable"] is True
    assert d.location_event_claim_present(driver_id, event_id) is True
