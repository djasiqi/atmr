"""Tests F-02 gps_spool — ACK/DLQ/replay Lua (Redis réel via docker)."""

from __future__ import annotations

import json
import os
import time

import pytest

redis = pytest.importorskip("redis")

REDIS_URL = os.getenv("WS_GPS_TEST_REDIS_URL", os.getenv("REDIS_URL", "redis://redis:6379/15"))


@pytest.fixture
def client():
    c = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        c.ping()
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"Redis indisponible: {exc}")
    # Isolation DB 15
    c.flushdb()
    import gps_spool as spool

    spool.configure_redis(c)
    spool.ensure_group(c)
    yield c
    c.flushdb()


def _admit(client, *, eid: str = "e1"):
    import gps_spool as spool
    from event_payload_hash import compute_event_payload_hash_from_point

    point = {
        "latitude": 46.2,
        "longitude": 6.1,
        "recorded_at": "2026-07-26T12:00:00.000Z",
        "location_event_id": eid,
        "location_mode": "mission_live",
    }
    phash, _ = compute_event_payload_hash_from_point(point)
    ok, sid = spool.admit(
        client, driver_id=7, company_id=10, point=point, event_payload_hash=phash
    )
    assert ok is True
    return sid


def test_ack_double_exec_counters_stable(client):
    import gps_spool as spool

    sid = _admit(client)
    # Simuler PEL : lire via groupe
    spool.read_batch(client, count=1)
    before_e = int(client.get(spool.STATS_PENDING_EVENTS) or 0)
    before_b = int(client.get(spool.STATS_PENDING_BYTES) or 0)
    n1 = spool.ack_batch(client, [sid])
    mid_e = int(client.get(spool.STATS_PENDING_EVENTS) or 0)
    n2 = spool.ack_batch(client, [sid])
    after_e = int(client.get(spool.STATS_PENDING_EVENTS) or 0)
    after_b = int(client.get(spool.STATS_PENDING_BYTES) or 0)
    assert n1 == 1
    assert n2 == 0
    assert mid_e == before_e - 1
    assert after_e == mid_e
    assert after_b <= before_b


def test_dlq_double_exec_one_entry(client):
    import gps_spool as spool

    sid = _admit(client)
    spool.read_batch(client, count=1)
    st1, d1 = spool.transfer_dlq(client, sid, reason="batch_payload_conflict", force=True)
    st2, d2 = spool.transfer_dlq(client, sid, reason="batch_payload_conflict", force=True)
    assert st1 == "ok"
    assert st2 == "already"
    assert d1 == d2
    assert int(client.xlen(spool.STREAM_DLQ)) == 1


def test_dlq_full_keeps_pending(client, monkeypatch):
    import gps_spool as spool

    monkeypatch.setattr(spool, "DLQ_MAX_EVENTS", 0)
    sid = _admit(client)
    spool.read_batch(client, count=1)
    st, detail = spool.transfer_dlq(client, sid, reason="validation", force=False)
    assert st == "full"
    assert detail == "dlq_full"
    assert int(client.xlen(spool.STREAM_PENDING)) == 1
    assert int(client.xlen(spool.STREAM_DLQ)) == 0


def test_replay_respects_deadline(client):
    import gps_spool as spool

    sid = _admit(client)
    spool.read_batch(client, count=1)
    st, dlq_id = spool.transfer_dlq(client, sid, reason="validation", force=True)
    assert st == "ok"
    entries = client.xrange(spool.STREAM_DLQ, dlq_id, dlq_id)
    assert entries
    fields = dict(entries[0][1])
    client.xdel(spool.STREAM_DLQ, dlq_id)
    pe = int(client.get(spool.STATS_DLQ_EVENTS) or 0)
    if pe > 0:
        client.decr(spool.STATS_DLQ_EVENTS)
    expired_id = client.xadd(
        spool.STREAM_DLQ,
        {
            "payload": fields.get("payload", "{}"),
            "driver_id": fields.get("driver_id", "7"),
            "company_id": fields.get("company_id", "10"),
            "event_payload_hash": fields.get("event_payload_hash", ""),
            "location_event_id": fields.get("location_event_id", "e1"),
            "first_spooled_at": str(time.time() - 90000),
            "replay_deadline": str(time.time() - 10),
            "source_stream_id": sid,
            "dlq_reason": "validation",
        },
    )
    client.incr(spool.STATS_DLQ_EVENTS)
    status, detail = spool.replay_dlq_entry(client, str(expired_id))
    assert status == "abort"
    assert detail == "deadline_exceeded"
    assert int(client.xlen(spool.STREAM_DLQ)) == 1


def test_replay_spool_full_keeps_dlq(client, monkeypatch):
    import gps_spool as spool

    sid = _admit(client)
    spool.read_batch(client, count=1)
    st, dlq_id = spool.transfer_dlq(client, sid, reason="validation", force=True)
    assert st == "ok"
    monkeypatch.setattr(spool, "SPOOL_MAX_EVENTS", 0)
    status, detail = spool.replay_dlq_entry(client, dlq_id)
    assert status == "abort"
    assert detail == "spool_full"
    assert int(client.xlen(spool.STREAM_DLQ)) == 1
