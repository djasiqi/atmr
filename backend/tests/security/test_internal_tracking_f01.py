"""Tests F-01 — ingestion GPS interne fail-closed."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import patch

import pytest

TOKEN = "a" * 32
TOKEN_NEXT = "b" * 32
INGEST_PATH = "/api/internal/tracking/ingest"


def _mock_persist_success():
    """Patches pour un ingest 200 sans DB réelle (F-02 sync_db)."""
    from unittest.mock import MagicMock

    from services.tracking.ingest_durability import BatchPersistResult

    def _persist(**kwargs):
        prep = kwargs["prepared"]
        eids = tuple(p.payload["location_event_id"] for p in prep.points)
        return BatchPersistResult(
            received=len(eids),
            persisted=len(eids),
            duplicates=0,
            batch_id=prep.batch_id,
            trace_id="t1",
            event_ids_persisted=eids,
            event_ids_duplicate=(),
        )

    mock_db = MagicMock()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=None)
    cm.__exit__ = MagicMock(return_value=None)
    mock_db.session.begin.return_value = cm
    return [
        patch(
            "routes.internal_tracking.persist_tracking_batch",
            side_effect=_persist,
        ),
        patch(
            "routes.internal_tracking.attempt_redis_canonical_repair", return_value=True
        ),
        patch("routes.internal_tracking.mark_repair_done_if_current"),
        patch("routes.internal_tracking.db", mock_db),
    ]


def _auth_headers(**extra: str) -> dict[str, str]:
    h = {
        "X-Internal-Token": TOKEN,
        "X-Internal-Service": "ws-service",
        "Content-Type": "application/json",
    }
    h.update(extra)
    return h


def _point(**overrides: Any) -> dict[str, Any]:
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    base = {
        "latitude": 46.5197,
        "longitude": 6.6323,
        "recorded_at": now,
        "location_mode": "mission_live",
        "location_event_id": "evt-unique-1",
    }
    base.update(overrides)
    return base


@pytest.fixture
def f01_env(monkeypatch):
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", TOKEN)
    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN_NEXT", "")
    monkeypatch.setenv("INTERNAL_SERVICE_AUDIENCE", "ws-service")
    monkeypatch.setenv("INTERNAL_TRACKING_INGEST_ENABLED", "true")
    monkeypatch.setenv("INTERNAL_TRACKING_IDEMPOTENCY_PENDING_TTL_SEC", "60")
    monkeypatch.setenv("INTERNAL_TRACKING_IDEMPOTENCY_DONE_TTL_SEC", "86400")
    monkeypatch.setenv("INTERNAL_TRACKING_DURABILITY_MODE", "sync_db")
    monkeypatch.setenv("KAFKA_PRODUCE_TIMEOUT_S", "1.5")
    monkeypatch.setenv("KAFKA_MAX_BLOCK_MS", "1000")


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
            if "DEL" in script:
                if store.get(key) == expected:
                    del store[key]
                    return 1
                return 0
            # mark_done
            if store.get(key) == expected:
                store[key] = "done"
                return 1
            return 0

        def delete(self, *keys):
            for k in keys:
                store.pop(k, None)

    return FakeRedis(), store


class TestBootValidation:
    def test_boot_missing_token(self, f01_env, monkeypatch):
        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "")
        from services.security.internal_service_auth import (
            validate_internal_service_token_for_boot,
        )

        with pytest.raises(RuntimeError, match="INTERNAL_SERVICE_TOKEN"):
            validate_internal_service_token_for_boot(config_name="production")

    def test_boot_short_token(self, f01_env, monkeypatch):
        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "short")
        from services.security.internal_service_auth import (
            validate_internal_service_token_for_boot,
        )

        with pytest.raises(RuntimeError, match="trop court"):
            validate_internal_service_token_for_boot(config_name="production")

    def test_boot_ok(self, f01_env):
        from services.security.internal_service_auth import (
            validate_internal_service_token_for_boot,
        )

        validate_internal_service_token_for_boot(config_name="production")


class TestAuthAudience:
    def test_anonymous_401(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        with (
            patch("routes.internal_tracking.redis_client", redis),
            patch(
                "services.tracking.ingest_idempotency._get_redis", return_value=redis
            ),
        ):
            resp = client.post(
                INGEST_PATH,
                data=json.dumps({"driver_id": 1, "points": [_point()]}),
                content_type="application/json",
            )
        assert resp.status_code == 401

    def test_bad_token_same_length_401(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        with patch("routes.internal_tracking.redis_client", redis):
            resp = client.post(
                INGEST_PATH,
                data=json.dumps({"driver_id": 1, "points": [_point()]}),
                headers=_auth_headers(**{"X-Internal-Token": "c" * 32}),
            )
        assert resp.status_code == 401
        assert resp.get_json()["error"] == "invalid_token"

    def test_audience_missing_401(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        headers = _auth_headers()
        del headers["X-Internal-Service"]
        with patch("routes.internal_tracking.redis_client", redis):
            resp = client.post(
                INGEST_PATH,
                data=json.dumps({"driver_id": 1, "points": [_point()]}),
                headers=headers,
            )
        assert resp.status_code == 401
        assert resp.get_json()["error"] == "invalid_audience"

    def test_audience_wrong_401(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        with patch("routes.internal_tracking.redis_client", redis):
            resp = client.post(
                INGEST_PATH,
                data=json.dumps({"driver_id": 1, "points": [_point()]}),
                headers=_auth_headers(**{"X-Internal-Service": "other-service"}),
            )
        assert resp.status_code == 401

    def test_audience_legacy_regex_like_401(self, client, f01_env, fake_redis):
        """Ancienne regex ^[a-z0-9]... acceptait d'autres valeurs — désormais 401."""
        redis, _ = fake_redis
        with patch("routes.internal_tracking.redis_client", redis):
            resp = client.post(
                INGEST_PATH,
                data=json.dumps({"driver_id": 1, "points": [_point()]}),
                headers=_auth_headers(**{"X-Internal-Service": "ws.service.extra"}),
            )
        assert resp.status_code == 401

    def test_dual_token_next_accepted(self, client, f01_env, monkeypatch, fake_redis):
        monkeypatch.setenv("INTERNAL_SERVICE_TOKEN_NEXT", TOKEN_NEXT)
        monkeypatch.setenv("INTERNAL_TRACKING_DURABILITY_MODE", "sync_db")
        redis, _ = fake_redis
        patches = [
            patch("routes.internal_tracking.redis_client", redis),
            patch(
                "services.tracking.ingest_idempotency._get_redis", return_value=redis
            ),
            patch(
                "routes.internal_tracking._resolve_driver_tenant",
                return_value=(10, None),
            ),
            *_mock_persist_success(),
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            resp = client.post(
                INGEST_PATH,
                data=json.dumps({"driver_id": 1, "points": [_point()]}),
                headers=_auth_headers(**{"X-Internal-Token": TOKEN_NEXT}),
            )
        assert resp.status_code == 200


class TestValidation:
    def _post(self, client, redis, body, headers=None):
        patches = [
            patch("routes.internal_tracking.redis_client", redis),
            patch(
                "services.tracking.ingest_idempotency._get_redis", return_value=redis
            ),
            patch(
                "routes.internal_tracking._resolve_driver_tenant",
                return_value=(10, None),
            ),
            *_mock_persist_success(),
        ]
        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
            patches[4],
            patches[5],
            patches[6],
        ):
            from unittest.mock import MagicMock

            enq = MagicMock()
            resp = client.post(
                INGEST_PATH,
                data=json.dumps(body),
                headers=headers or _auth_headers(),
            )
            return resp, enq

    def test_nan_rejected(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        resp, enq = self._post(
            client,
            redis,
            {"driver_id": 1, "points": [_point(latitude="nan")]},
        )
        assert resp.status_code == 400
        enq.assert_not_called()

    def test_bool_coord_rejected(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        resp, enq = self._post(
            client,
            redis,
            {"driver_id": 1, "points": [_point(latitude=True)]},
        )
        assert resp.status_code == 400
        enq.assert_not_called()

    def test_timestamp_too_old(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        old = (datetime.now(UTC) - timedelta(hours=25)).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3] + "Z"
        resp, enq = self._post(
            client,
            redis,
            {"driver_id": 1, "points": [_point(recorded_at=old)]},
        )
        assert resp.status_code == 400
        enq.assert_not_called()

    def test_batch_partial_invalid_atomic(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        resp, enq = self._post(
            client,
            redis,
            {
                "driver_id": 1,
                "points": [
                    _point(location_event_id="a"),
                    _point(location_event_id="b", latitude=999),
                ],
            },
        )
        assert resp.status_code == 400
        enq.assert_not_called()

    def test_header_event_id_batch_rejected(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        resp, enq = self._post(
            client,
            redis,
            {
                "driver_id": 1,
                "points": [
                    _point(location_event_id="a"),
                    _point(location_event_id="b"),
                ],
            },
            headers=_auth_headers(**{"X-Location-Event-ID": "shared"}),
        )
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "header_event_id_not_allowed_for_batch"
        enq.assert_not_called()

    def test_duplicate_ids_in_batch(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        resp, enq = self._post(
            client,
            redis,
            {
                "driver_id": 1,
                "points": [
                    _point(location_event_id="same"),
                    _point(location_event_id="same", latitude=46.52),
                ],
            },
        )
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "duplicate_location_event_id_in_batch"
        enq.assert_not_called()

    def test_ingest_disabled_503(self, client, f01_env, monkeypatch, fake_redis):
        monkeypatch.setenv("INTERNAL_TRACKING_INGEST_ENABLED", "false")
        redis, store = fake_redis
        with patch("routes.internal_tracking.redis_client", redis):
            resp = client.post(
                INGEST_PATH,
                data=json.dumps({"driver_id": 1, "points": [_point()]}),
                headers=_auth_headers(),
            )
        assert resp.status_code == 503
        assert resp.get_json()["error"] == "ingest_disabled"
        assert store == {}

    def test_driver_inactive_403(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        with (
            patch("routes.internal_tracking.redis_client", redis),
            patch(
                "routes.internal_tracking._resolve_driver_tenant",
                return_value=(None, "driver_inactive"),
            ),
        ):
            resp = client.post(
                INGEST_PATH,
                data=json.dumps({"driver_id": 1, "points": [_point()]}),
                headers=_auth_headers(),
            )
        assert resp.status_code == 403

    def test_happy_path_and_duplicate(self, client, f01_env, fake_redis):
        redis, _store = fake_redis
        from services.tracking.ingest_durability import BatchPersistResult

        calls = {"n": 0}

        def _persist(**kwargs):
            calls["n"] += 1
            prep = kwargs["prepared"]
            if calls["n"] == 1:
                return BatchPersistResult(
                    received=1,
                    persisted=1,
                    duplicates=0,
                    batch_id=prep.batch_id,
                    trace_id="t1",
                    event_ids_persisted=(prep.points[0].payload["location_event_id"],),
                    event_ids_duplicate=(),
                )
            return BatchPersistResult(
                received=1,
                persisted=0,
                duplicates=1,
                batch_id=prep.batch_id,
                trace_id="t2",
                event_ids_persisted=(),
                event_ids_duplicate=(prep.points[0].payload["location_event_id"],),
            )

        with (
            patch("routes.internal_tracking.redis_client", redis),
            patch(
                "services.tracking.ingest_idempotency._get_redis", return_value=redis
            ),
            patch(
                "routes.internal_tracking._resolve_driver_tenant",
                return_value=(10, None),
            ),
            patch(
                "routes.internal_tracking.persist_tracking_batch",
                side_effect=_persist,
            ),
            patch(
                "routes.internal_tracking.attempt_redis_canonical_repair",
                return_value=True,
            ),
            patch("routes.internal_tracking.mark_repair_done_if_current"),
            patch("routes.internal_tracking.db") as mock_db,
        ):
            mock_db.session.begin.return_value.__enter__ = lambda s: s
            mock_db.session.begin.return_value.__exit__ = lambda *a: None
            body = {"driver_id": 1, "points": [_point(location_event_id="dup-1")]}
            r1 = client.post(
                INGEST_PATH, data=json.dumps(body), headers=_auth_headers()
            )
            r2 = client.post(
                INGEST_PATH, data=json.dumps(body), headers=_auth_headers()
            )
        assert r1.status_code == 200
        assert r1.get_json()["persisted"] == 1
        assert r1.get_json()["durability"] == "postgres_committed"
        assert r2.status_code == 200
        assert r2.get_json()["duplicates"] == 1

    def test_persist_fail_503(self, client, f01_env, fake_redis):
        redis, _store = fake_redis
        with (
            patch("routes.internal_tracking.redis_client", redis),
            patch(
                "services.tracking.ingest_idempotency._get_redis", return_value=redis
            ),
            patch(
                "routes.internal_tracking._resolve_driver_tenant",
                return_value=(10, None),
            ),
            patch(
                "routes.internal_tracking.persist_tracking_batch",
                side_effect=RuntimeError("db down"),
            ),
            patch("routes.internal_tracking.db") as mock_db,
        ):
            mock_db.session.begin.return_value.__enter__ = lambda s: s
            mock_db.session.begin.return_value.__exit__ = lambda *a: True

            # begin context raising
            class _CM:
                def __enter__(self):
                    raise RuntimeError("db down")

                def __exit__(self, *a):
                    return False

            mock_db.session.begin.return_value = _CM()
            resp = client.post(
                INGEST_PATH,
                data=json.dumps(
                    {"driver_id": 1, "points": [_point(location_event_id="kfail")]}
                ),
                headers=_auth_headers(),
            )
        assert resp.status_code == 503
        assert resp.get_json()["error"] == "ingest_persistence_failed"

    def test_redis_down_continues_to_pg(self, client, f01_env):
        """F-02 : Redis KO n'empêche pas sync_db (accélérateur seulement)."""
        patches = _mock_persist_success()
        with (
            patch("routes.internal_tracking.redis_client", None),
            patch(
                "routes.internal_tracking._resolve_driver_tenant",
                return_value=(10, None),
            ),
            patches[0],
            patches[1],
            patches[2],
            patches[3],
        ):
            resp = client.post(
                INGEST_PATH,
                data=json.dumps({"driver_id": 1, "points": [_point()]}),
                headers=_auth_headers(),
            )
        assert resp.status_code == 200
        assert resp.get_json()["durability"] == "postgres_committed"

    def test_body_too_large_413(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        big = "x" * (65 * 1024)
        with patch("routes.internal_tracking.redis_client", redis):
            resp = client.post(
                INGEST_PATH,
                data=big,
                headers={
                    **_auth_headers(),
                    "Content-Type": "application/json",
                    "Content-Length": str(len(big)),
                },
            )
        assert resp.status_code == 413

    def test_batch_too_large_400(self, client, f01_env, fake_redis):
        redis, _ = fake_redis
        points = [
            _point(location_event_id=f"e{i}", latitude=46.0 + i * 0.001)
            for i in range(51)
        ]
        resp, enq = self._post(client, redis, {"driver_id": 1, "points": points})
        assert resp.status_code == 400
        enq.assert_not_called()


class TestIdempotencyLua:
    def test_mark_done_nonce_mismatch(self, fake_redis):
        redis, store = fake_redis
        from services.tracking.ingest_idempotency import (
            mark_done,
            redis_key_for_event,
        )

        key = redis_key_for_event(driver_id=1, location_event_id="e1")
        store[key] = "pending:other-nonce"
        with patch(
            "services.tracking.ingest_idempotency._get_redis", return_value=redis
        ):
            ok = mark_done(driver_id=1, location_event_id="e1", nonce="my-nonce")
        assert ok is False
        assert store[key] == "pending:other-nonce"


class TestRateLimitPrincipal:
    def test_principal_constant(self):
        from services.security.internal_service_auth import rate_limit_principal

        assert rate_limit_principal() == "ws-service"


class TestLocationEventIdCanonical:
    def test_same_instant_same_id(self):
        from services.tracking.location_event_id import resolve_location_event_id

        a = resolve_location_event_id(
            driver_id=1,
            latitude=46.5,
            longitude=6.6,
            recorded_at="2024-01-01T12:00:00+00:00",
        )
        b = resolve_location_event_id(
            driver_id=1,
            latitude=46.5,
            longitude=6.6,
            recorded_at="2024-01-01T12:00:00Z",
        )
        assert a == b
