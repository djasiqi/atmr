"""Tests F-02 — ACK durable, batch_id, 409 conflict, persist sync_db."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from services.tracking.event_payload_hash import (
    compute_batch_id,
    compute_event_payload_hash,
)
from services.tracking.ingest_durability import (
    BatchPersistResult,
    PayloadConflictError,
    prepare_tracking_batch,
)

TOKEN = "a" * 32
INGEST_PATH = "/api/internal/tracking/ingest"


def _auth_headers() -> dict[str, str]:
    return {
        "X-Internal-Token": TOKEN,
        "X-Internal-Service": "ws-service",
        "Content-Type": "application/json",
    }


def _point(**overrides):
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    base = {
        "latitude": 46.5197,
        "longitude": 6.6323,
        "recorded_at": now,
        "location_mode": "mission_live",
        "location_event_id": "evt-f02-1",
    }
    base.update(overrides)
    return base


@pytest.fixture
def f02_env(monkeypatch):
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", TOKEN)
    monkeypatch.setenv("INTERNAL_SERVICE_AUDIENCE", "ws-service")
    monkeypatch.setenv("INTERNAL_TRACKING_INGEST_ENABLED", "true")
    monkeypatch.setenv("INTERNAL_TRACKING_DURABILITY_MODE", "sync_db")


@pytest.fixture
def fake_redis():
    store: dict[str, str] = {}

    class FakeRedis:
        def ping(self):
            return True

        def set(self, key, value, nx=False, ex=None):
            if nx and key in store:
                return False
            store[key] = value
            return True

        def get(self, key):
            return store.get(key)

        def eval(self, script, numkeys, *args):
            key = args[0]
            expected = args[1]
            if "DEL" in script and "done" not in script.lower():
                if store.get(key) == expected:
                    del store[key]
                    return 1
                return 0
            if store.get(key) == expected:
                store[key] = "done"
                return 1
            return 0

    return FakeRedis(), store


def _persist_patches(side_effect=None):
    def _default(**kwargs):
        prep = kwargs["prepared"]
        eids = tuple(p.payload["location_event_id"] for p in prep.points)
        return BatchPersistResult(
            received=len(eids),
            persisted=len(eids),
            duplicates=0,
            batch_id=prep.batch_id,
            trace_id="tr",
            event_ids_persisted=eids,
            event_ids_duplicate=(),
        )

    mock_db = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__ = MagicMock(return_value=None)
    mock_db.session.begin.return_value = cm
    return (
        patch(
            "routes.internal_tracking.persist_tracking_batch",
            side_effect=side_effect or _default,
        ),
        patch(
            "routes.internal_tracking.attempt_redis_canonical_repair", return_value=True
        ),
        patch("routes.internal_tracking.mark_repair_done_if_current"),
        patch("routes.internal_tracking.db", mock_db),
    )


def test_batch_id_mismatch_400(client, f02_env, fake_redis):
    redis, _ = fake_redis
    p0, p1, p2, p3 = _persist_patches()
    with (
        patch("routes.internal_tracking.redis_client", redis),
        patch("services.tracking.ingest_idempotency._get_redis", return_value=redis),
        patch(
            "routes.internal_tracking._resolve_driver_tenant",
            return_value=(10, None),
        ),
        p0,
        p1,
        p2,
        p3,
    ):
        resp = client.post(
            INGEST_PATH,
            data=json.dumps(
                {
                    "driver_id": 1,
                    "batch_id": "deadbeef" * 8,
                    "points": [_point()],
                }
            ),
            headers=_auth_headers(),
        )
    assert resp.status_code == 400
    assert resp.get_json()["error"] == "batch_id_mismatch"


def test_409_payload_conflict(client, f02_env, fake_redis):
    redis, _ = fake_redis

    def _conflict(**kwargs):
        raise PayloadConflictError("event_id_payload_conflict", ["evt-f02-1"])

    p0, p1, p2, p3 = _persist_patches(side_effect=_conflict)
    with (
        patch("routes.internal_tracking.redis_client", redis),
        patch("services.tracking.ingest_idempotency._get_redis", return_value=redis),
        patch(
            "routes.internal_tracking._resolve_driver_tenant",
            return_value=(10, None),
        ),
        p0,
        p1,
        p2,
        p3,
    ):
        resp = client.post(
            INGEST_PATH,
            data=json.dumps({"driver_id": 1, "points": [_point()]}),
            headers=_auth_headers(),
        )
    assert resp.status_code == 409
    body = resp.get_json()
    assert body["error_code"] == "event_id_payload_conflict"
    assert body["conflicting_event_ids"] == ["evt-f02-1"]
    assert body["durability"] == "none"


def test_200_response_shape(client, f02_env, fake_redis):
    redis, _ = fake_redis
    p0, p1, p2, p3 = _persist_patches()
    with (
        patch("routes.internal_tracking.redis_client", redis),
        patch("services.tracking.ingest_idempotency._get_redis", return_value=redis),
        patch(
            "routes.internal_tracking._resolve_driver_tenant",
            return_value=(10, None),
        ),
        p0,
        p1,
        p2,
        p3,
    ):
        pt = _point()
        prepared = prepare_tracking_batch(driver_id=1, company_id=10, points=[pt])
        resp = client.post(
            INGEST_PATH,
            data=json.dumps(
                {
                    "driver_id": 1,
                    "company_id": 10,
                    "batch_id": prepared.batch_id,
                    "points": [pt],
                }
            ),
            headers=_auth_headers(),
        )
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert body["durability"] == "postgres_committed"
    assert body["received"] == body["persisted"] + body["duplicates"]
    assert body["batch_id"] == prepared.batch_id


def test_prepare_batch_id_matches_hash():
    pt = _point(location_event_id="x1")
    phash, _ = compute_event_payload_hash(
        location_event_id="x1",
        recorded_at=pt["recorded_at"],
        latitude=pt["latitude"],
        longitude=pt["longitude"],
        location_mode="mission_live",
    )
    expected = compute_batch_id(driver_id=7, company_id=3, events=[("x1", phash)])
    prepared = prepare_tracking_batch(
        driver_id=7, company_id=3, points=[pt], client_batch_id=expected
    )
    assert prepared.batch_id == expected
