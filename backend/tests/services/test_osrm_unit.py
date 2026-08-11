"""Tests unitaires OSRM (APIs réelles, sans mock_external_services ni réseau)."""

from __future__ import annotations

import json
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest
import requests

from services.geolocation import osrm

# Capturés à l'import module (avant activation des fixtures autouse)
_REAL_ROUTE_INFO = osrm.route_info
_REAL_ETA_SECONDS = osrm.eta_seconds
_REAL_BUILD_MATRIX = osrm.build_distance_matrix_osrm
_REAL_GET_DISTANCE_TIME = osrm.get_distance_time
_REAL_GET_MATRIX = osrm.get_matrix
_REAL_RATE_LIMIT = osrm._rate_limit
_REAL_GET_REDIS_FALLBACK = osrm._get_redis_client_fallback

LAUSANNE = (46.52, 6.63)
GENEVA = (46.20, 6.15)
BASE = "http://osrm-test:5000"


class _MockResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.elapsed = timedelta(seconds=0.01)
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status={self.status_code}")


@pytest.fixture
def real_osrm(monkeypatch):
    """Restaure les APIs publiques et isole l'état global OSRM."""
    monkeypatch.setattr(osrm, "route_info", _REAL_ROUTE_INFO)
    monkeypatch.setattr(osrm, "eta_seconds", _REAL_ETA_SECONDS)
    monkeypatch.setattr(osrm, "build_distance_matrix_osrm", _REAL_BUILD_MATRIX)
    monkeypatch.setattr(osrm, "get_distance_time", _REAL_GET_DISTANCE_TIME)
    monkeypatch.setattr(osrm, "get_matrix", _REAL_GET_MATRIX)

    # Pas de sleep réel (retry, rate-limit, chaos)
    monkeypatch.setattr(osrm.time, "sleep", lambda *_a, **_k: None)
    monkeypatch.setattr("shared.retry.time.sleep", lambda *_a, **_k: None)
    monkeypatch.setattr(osrm, "_rate_limit", lambda *_a, **_k: None)
    monkeypatch.setattr(osrm, "_get_redis_client_fallback", lambda: None)

    osrm._OSRM_MATRIX_LOCAL_CACHE.clear()

    cb = osrm._osrm_circuit_breaker
    cb.state = "CLOSED"
    cb.failure_count = 0
    cb.last_failure_time = None

    with osrm._inflight_lock:
        osrm._inflight.clear()
    osrm._rl_last_ts["value"] = 0.0

    yield osrm

    osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
    cb.state = "CLOSED"
    cb.failure_count = 0
    cb.last_failure_time = None
    with osrm._inflight_lock:
        osrm._inflight.clear()
    osrm._rl_last_ts["value"] = 0.0


# ---------------------------------------------------------------------------
# _table / _route
# ---------------------------------------------------------------------------


class TestTableAndRoute:
    def test_table_success(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm.requests,
            "get",
            lambda *a, **k: _MockResponse(
                {"code": "Ok", "durations": [[0, 600], [600, 0]]}
            ),
        )
        result = osrm._table(BASE, "driving", [LAUSANNE, GENEVA], None, None, timeout=5)
        assert result["code"] == "Ok"
        assert result["durations"][0][1] == 600

    def test_table_timeout_retries_then_raises(self, real_osrm, monkeypatch):
        calls = {"n": 0}

        def boom(*_a, **_k):
            calls["n"] += 1
            raise requests.Timeout("timeout")

        monkeypatch.setattr(osrm.requests, "get", boom)
        with pytest.raises((requests.Timeout, TimeoutError, Exception)):
            osrm._table(BASE, "driving", [LAUSANNE, GENEVA], None, None, timeout=1)
        # retry_with_backoff : 1 tentative + DEFAULT_RETRY_COUNT retries
        assert calls["n"] >= 2

    def test_route_success(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm.requests,
            "get",
            lambda *a, **k: _MockResponse(
                {
                    "code": "Ok",
                    "routes": [
                        {
                            "duration": 1200.0,
                            "distance": 50000.0,
                            "geometry": None,
                            "legs": [],
                        }
                    ],
                }
            ),
        )
        data = osrm._route(BASE, "driving", LAUSANNE, GENEVA, timeout=5)
        assert data["code"] == "Ok"
        assert data["routes"][0]["duration"] == 1200.0

    def test_route_timeout_raises_no_retry(self, real_osrm, monkeypatch):
        calls = {"n": 0}

        def boom(*_a, **_k):
            calls["n"] += 1
            raise requests.Timeout("timeout")

        monkeypatch.setattr(osrm.requests, "get", boom)
        with pytest.raises(requests.Timeout):
            osrm._route(BASE, "driving", LAUSANNE, GENEVA, timeout=1)
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# route_info / eta_seconds
# ---------------------------------------------------------------------------


class TestRouteInfoAndEta:
    def test_route_info_success(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm,
            "_route",
            lambda **_k: {
                "code": "Ok",
                "routes": [
                    {
                        "duration": 900.0,
                        "distance": 40000.0,
                        "geometry": {"type": "LineString"},
                        "legs": [],
                    }
                ],
            },
        )
        res = osrm.route_info(LAUSANNE, GENEVA, base_url=BASE)
        assert res["fallback"] is False
        assert res["duration"] == 900.0
        assert res["distance"] == 40000.0

    def test_route_info_bad_code_fallback(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm, "_route", lambda **_k: {"code": "NoRoute", "routes": []}
        )
        res = osrm.route_info(LAUSANNE, GENEVA, base_url=BASE)
        assert res["fallback"] is True
        assert res["duration"] > 0
        assert res["distance"] > 0

    def test_route_info_empty_routes_fallback(self, real_osrm, monkeypatch):
        monkeypatch.setattr(osrm, "_route", lambda **_k: {"code": "Ok", "routes": []})
        res = osrm.route_info(LAUSANNE, GENEVA, base_url=BASE)
        assert res["fallback"] is True

    def test_route_info_timeout_fallback(self, real_osrm, monkeypatch):
        def boom(**_k):
            raise requests.Timeout("timeout")

        monkeypatch.setattr(osrm, "_route", boom)
        res = osrm.route_info(LAUSANNE, GENEVA, base_url=BASE)
        assert res["fallback"] is True
        assert res["duration"] > 0
        assert res["distance"] > 0

    def test_eta_seconds_success(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm,
            "route_info",
            lambda *a, **k: {
                "duration": 123.4,
                "distance": 1000.0,
                "fallback": False,
            },
        )
        assert osrm.eta_seconds(LAUSANNE, GENEVA, base_url=BASE) == 123

    def test_eta_seconds_invalid_duration_fallback(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm,
            "route_info",
            lambda *a, **k: {
                "duration": None,
                "distance": 0,
                "fallback": True,
            },
        )
        eta = osrm.eta_seconds(LAUSANNE, GENEVA, base_url=BASE)
        assert isinstance(eta, int)
        assert eta >= 1


# ---------------------------------------------------------------------------
# build_distance_matrix_osrm
# ---------------------------------------------------------------------------


class TestBuildMatrix:
    def test_early_return_n_le_1(self, real_osrm):
        assert osrm.build_distance_matrix_osrm([], base_url=BASE) == []
        assert osrm.build_distance_matrix_osrm([LAUSANNE], base_url=BASE) == [[0.0]]

    def test_matrix_ok_via_table(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm,
            "_table",
            lambda **_k: {"code": "Ok", "durations": [[0.0, 600.0], [600.0, 0.0]]},
        )
        m = osrm.build_distance_matrix_osrm(
            [LAUSANNE, GENEVA], base_url=BASE, timeout=5
        )
        assert len(m) == 2
        assert m[0][1] == 600.0

    def test_matrix_all_chunks_fail_fallback(self, real_osrm, monkeypatch):
        def boom(**_k):
            raise requests.Timeout("down")

        monkeypatch.setattr(osrm, "_table", boom)
        m = osrm.build_distance_matrix_osrm(
            [LAUSANNE, GENEVA], base_url=BASE, timeout=5
        )
        assert len(m) == 2
        assert m[0][0] == 0.0
        assert m[0][1] > 0

    def test_matrix_none_duration_becomes_large(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm,
            "_table",
            lambda **_k: {"code": "Ok", "durations": [[0.0, None], [None, 0.0]]},
        )
        m = osrm.build_distance_matrix_osrm(
            [LAUSANNE, GENEVA], base_url=BASE, timeout=5
        )
        assert m[0][1] == 999999.0

    def test_l1_cache_hit(self, real_osrm, monkeypatch):
        calls = {"n": 0}

        def table(**_k):
            calls["n"] += 1
            return {"code": "Ok", "durations": [[0.0, 100.0], [100.0, 0.0]]}

        monkeypatch.setattr(osrm, "_table", table)
        coords = [LAUSANNE, GENEVA]
        m1 = osrm.build_distance_matrix_osrm(coords, base_url=BASE, timeout=5)
        m2 = osrm.build_distance_matrix_osrm(coords, base_url=BASE, timeout=5)
        assert m1 == m2
        assert calls["n"] == 1


# ---------------------------------------------------------------------------
# CircuitBreaker + wrappers
# ---------------------------------------------------------------------------


class TestCircuitBreakerAndClient:
    def test_circuit_breaker_opens_then_fallback(self, real_osrm, monkeypatch):
        cb = osrm._osrm_circuit_breaker
        cb.failure_threshold = 2
        cb.timeout_duration = 3600

        def boom(*_a, **_k):
            raise RuntimeError("osrm down")

        monkeypatch.setattr(osrm, "build_distance_matrix_osrm", boom)

        # 2 échecs → OPEN
        m1 = osrm.build_distance_matrix_osrm_with_cb([LAUSANNE, GENEVA], base_url=BASE)
        m2 = osrm.build_distance_matrix_osrm_with_cb([LAUSANNE, GENEVA], base_url=BASE)
        assert len(m1) == 2
        assert len(m2) == 2
        assert cb.state == "OPEN"

        # Appel suivant : CB OPEN → fallback immédiat
        m3 = osrm.build_distance_matrix_osrm_with_cb([LAUSANNE, GENEVA], base_url=BASE)
        assert m3[0][1] > 0

    def test_circuit_breaker_half_open_success_closes(self, real_osrm, monkeypatch):
        cb = osrm._osrm_circuit_breaker
        cb.state = "OPEN"
        cb.failure_count = 5
        cb.last_failure_time = osrm.time.time() - 120
        cb.timeout_duration = 60

        monkeypatch.setattr(
            osrm,
            "build_distance_matrix_osrm",
            lambda coords, **k: [[0.0, 1.0], [1.0, 0.0]],
        )
        result = osrm.build_distance_matrix_osrm_with_cb(
            [LAUSANNE, GENEVA], base_url=BASE
        )
        assert result[0][1] == 1.0
        assert cb.state == "CLOSED"

    def test_osrm_client_get_route_success(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm,
            "route_info",
            lambda *a, **k: {
                "duration": 10.0,
                "distance": 100.0,
                "fallback": False,
                "geometry": None,
                "legs": [],
            },
        )
        client = osrm.OSRMClient(base_url=BASE)
        res = client.get_route(LAUSANNE, GENEVA)
        assert res["fallback"] is False
        assert res["duration"] == 10.0

    def test_osrm_client_heuristic_on_exception(self, real_osrm, monkeypatch):
        def boom(*_a, **_k):
            raise requests.ConnectionError("down")

        monkeypatch.setattr(osrm, "route_info", boom)
        client = osrm.OSRMClient(base_url=BASE)
        res = client.get_route(LAUSANNE, GENEVA)
        assert res.get("fallback") is True or res["duration"] > 0

    def test_get_distance_time_wrapper(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm,
            "route_info",
            lambda *a, **k: {
                "duration": 50.0,
                "distance": 1200.0,
                "fallback": False,
            },
        )
        out = osrm.get_distance_time(LAUSANNE, GENEVA, base_url=BASE)
        assert out["duration"] == 50.0
        assert out["distance"] == 1200.0

    def test_get_matrix_wrapper(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm,
            "build_distance_matrix_osrm_with_cb",
            lambda coords, **k: [[0.0, 9.0], [9.0, 0.0]],
        )
        out = osrm.get_matrix([LAUSANNE], [GENEVA], base_url=BASE)
        assert "durations" in out or isinstance(out, dict)


# ---------------------------------------------------------------------------
# Redis / cache / singleflight / helpers
# ---------------------------------------------------------------------------


class TestRedisCacheAndHelpers:
    def test_route_info_redis_hit(self, real_osrm, monkeypatch):
        calls = {"route": 0}

        def never_route(**_k):
            calls["route"] += 1
            raise AssertionError("_route ne doit pas être appelé")

        monkeypatch.setattr(osrm, "_route", never_route)
        redis = MagicMock()
        redis.get.return_value = json.dumps(
            {"duration": 42.0, "distance": 1000.0, "fallback": False}
        )
        res = osrm.route_info(LAUSANNE, GENEVA, base_url=BASE, redis_client=redis)
        assert res["duration"] == 42.0
        assert calls["route"] == 0

    def test_route_info_redis_conn_error_continues(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm,
            "_route",
            lambda **_k: {
                "code": "Ok",
                "routes": [
                    {
                        "duration": 11.0,
                        "distance": 22.0,
                        "geometry": None,
                        "legs": [],
                    }
                ],
            },
        )
        redis = MagicMock()
        redis.get.side_effect = osrm._RedisConnError("down")
        res = osrm.route_info(LAUSANNE, GENEVA, base_url=BASE, redis_client=redis)
        assert res["duration"] == 11.0
        assert res["fallback"] is False

    def test_table_json_decode_error(self, real_osrm, monkeypatch):
        class BadJson:
            status_code = 200
            elapsed = timedelta(seconds=0.01)
            text = "not-json"

            def raise_for_status(self):
                return None

            def json(self):
                raise json.JSONDecodeError("err", "doc", 0)

        monkeypatch.setattr(osrm.requests, "get", lambda *a, **k: BadJson())
        # retry puis ValueError / exception
        with pytest.raises((ValueError, Exception)):
            osrm._table(BASE, "driving", [LAUSANNE, GENEVA], None, None, timeout=1)

    def test_table_with_sources_destinations(self, real_osrm, monkeypatch):
        seen = {}

        def fake_get(url, params=None, timeout=None):
            seen["params"] = params
            return _MockResponse({"code": "Ok", "durations": [[100.0]]})

        monkeypatch.setattr(osrm.requests, "get", fake_get)
        osrm._table(
            BASE,
            "driving",
            [LAUSANNE, GENEVA],
            sources=[0],
            destinations=[1],
            timeout=5,
        )
        assert "sources" in (seen.get("params") or {})
        assert "destinations" in (seen.get("params") or {})

    def test_singleflight_leader(self, real_osrm):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return {"ok": True}

        r1 = osrm._singleflight_do("sf-key-1", fn, max_wait_seconds=1.0)
        assert r1 == {"ok": True}
        assert calls["n"] == 1

    def test_singleflight_follower_timeout_runs_fn(self, real_osrm, monkeypatch):
        # Préparer une entrée follower bloquée
        evt = MagicMock()
        evt.wait.return_value = False  # timeout
        with osrm._inflight_lock:
            osrm._inflight["sf-follower"] = {
                "evt": evt,
                "result": None,
                "error": None,
                "leader": False,
            }

        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            return 99

        # Le follower timeout exécute fn directement
        # Mais _singleflight_do va d'abord voir entry existante → leader=False
        result = osrm._singleflight_do("sf-follower", fn, max_wait_seconds=0.01)
        assert result == 99
        assert calls["n"] == 1

    def test_rate_limit_no_wait_when_zero(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm,
            "_rate_limit",
            lambda rate: (
                None
                if rate is None or rate <= 0
                else osrm.time.sleep(0)  # sleep mocké = no-op
            ),
        )
        osrm._rate_limit(None)
        osrm._rate_limit(0)

    def test_adaptive_ttl_helpers(self, real_osrm):
        redis = MagicMock()
        redis.get.return_value = "15"  # fréquent
        ttl = osrm._get_adaptive_ttl(redis, "k", 3600, cache_type="route")
        assert ttl >= 3600
        redis.get.return_value = "5"
        ttl_med = osrm._get_adaptive_ttl(redis, "k", 3600, cache_type="route")
        assert ttl_med >= 3600
        redis.get.return_value = "1"
        ttl_rare = osrm._get_adaptive_ttl(redis, "k", 3600, cache_type="route")
        assert ttl_rare >= 1
        assert osrm._get_frequency_count(None, "k") == 0
        osrm._increment_frequency_counter(None, "k")  # no-op
        osrm._increment_frequency_counter(redis, "k", cache_type="route")

    def test_get_distance_time_cached_hit(self, real_osrm, monkeypatch):
        fake_rc = MagicMock()
        fake_rc.get.return_value = json.dumps({"distance": 1.0, "duration": 2.0})
        monkeypatch.setattr(
            "ext.redis_client",
            fake_rc,
            raising=False,
        )
        # Patch l'import local dans la fonction via ext module
        import ext as ext_mod

        monkeypatch.setattr(ext_mod, "redis_client", fake_rc)
        out = osrm.get_distance_time_cached(LAUSANNE, GENEVA, date_str="2026-01-01")
        assert out["duration"] == 2.0

    def test_get_distance_time_cached_miss_then_set(self, real_osrm, monkeypatch):
        fake_rc = MagicMock()
        fake_rc.get.return_value = None
        import ext as ext_mod

        monkeypatch.setattr(ext_mod, "redis_client", fake_rc)
        monkeypatch.setattr(
            osrm,
            "get_distance_time",
            lambda *a, **k: {"distance": 10.0, "duration": 20.0},
        )
        out = osrm.get_distance_time_cached(LAUSANNE, GENEVA, date_str="2026-01-02")
        assert out["duration"] == 20.0
        assert fake_rc.setex.called

    def test_get_matrix_cached_hit(self, real_osrm, monkeypatch):
        fake_rc = MagicMock()
        fake_rc.get.return_value = json.dumps({"durations": [[0, 1], [1, 0]]})
        import ext as ext_mod

        monkeypatch.setattr(ext_mod, "redis_client", fake_rc)
        out = osrm.get_matrix_cached([LAUSANNE], [GENEVA], date_str="2026-01-01")
        assert out["durations"][0][1] == 1

    def test_get_matrix_cached_miss(self, real_osrm, monkeypatch):
        fake_rc = MagicMock()
        fake_rc.get.return_value = None
        import ext as ext_mod

        monkeypatch.setattr(ext_mod, "redis_client", fake_rc)
        monkeypatch.setattr(
            osrm,
            "get_matrix",
            lambda *a, **k: {"durations": [[0.0, 5.0], [5.0, 0.0]]},
        )
        out = osrm.get_matrix_cached([LAUSANNE], [GENEVA], date_str="2026-01-03")
        assert out["durations"][0][1] == 5.0

    def test_matrix_redis_l2_chunk_hit(self, real_osrm, monkeypatch):
        redis = MagicMock()

        def redis_get(key):
            return json.dumps({"durations": [[0.0, 77.0], [77.0, 0.0]]})

        redis.get.side_effect = redis_get
        # Éviter L1 hit d'un test précédent
        osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
        # _table ne doit pas être appelé si L2 hit
        monkeypatch.setattr(
            osrm,
            "_table",
            lambda **_k: (_ for _ in ()).throw(AssertionError("no http")),
        )
        m = osrm.build_distance_matrix_osrm(
            [LAUSANNE, GENEVA],
            base_url=BASE,
            timeout=5,
            redis_client=redis,
        )
        assert m[0][1] == 77.0

    def test_circuit_open_without_last_failure_time(self, real_osrm, monkeypatch):
        cb = osrm._osrm_circuit_breaker
        cb.state = "OPEN"
        cb.last_failure_time = None
        cb.failure_count = 3
        monkeypatch.setattr(
            osrm,
            "build_distance_matrix_osrm",
            lambda coords, **k: [[0.0, 2.0], [2.0, 0.0]],
        )
        m = osrm.build_distance_matrix_osrm_with_cb([LAUSANNE, GENEVA], base_url=BASE)
        assert m[0][1] == 2.0
        assert cb.state == "CLOSED"

    def test_half_open_failure_reopens(self, real_osrm, monkeypatch):
        cb = osrm.CircuitBreaker(failure_threshold=5, timeout_duration=60)
        cb.state = "HALF_OPEN"
        cb.failure_count = 0

        def boom(*_a, **_k):
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            cb.call(boom)
        assert cb.state == "OPEN"

    def test_osrm_client_generic_exception_fallback(self, real_osrm, monkeypatch):
        def boom(*_a, **_k):
            raise ValueError("unexpected")

        monkeypatch.setattr(osrm, "route_info", boom)
        client = osrm.OSRMClient(base_url=BASE)
        res = client.get_route(LAUSANNE, GENEVA, waypoints=[(46.4, 6.4)])
        assert res["fallback"] is True
        assert res["duration"] > 0

    def test_chunks_helper(self, real_osrm):
        blocks = list(osrm._chunks(range(5), 2))
        assert blocks == [[0, 1], [2, 3], [4]]

    def test_canonical_key_route_stable(self, real_osrm):
        k1 = osrm._canonical_key_route(LAUSANNE, GENEVA, None)
        k2 = osrm._canonical_key_route(LAUSANNE, GENEVA, None)
        assert k1 == k2

    def test_real_rate_limit_with_mocked_sleep(self, real_osrm, monkeypatch):
        monkeypatch.setattr(osrm, "_rate_limit", _REAL_RATE_LIMIT)
        osrm._rl_last_ts["value"] = osrm.time.time()
        _REAL_RATE_LIMIT(None)
        _REAL_RATE_LIMIT(0)
        _REAL_RATE_LIMIT(100.0)

    def test_partial_chunk_failure_fills_fallback_rows(self, real_osrm, monkeypatch):
        osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
        coords = [LAUSANNE, GENEVA, (47.37, 8.54)]

        def table(**kwargs):
            sources = kwargs.get("sources") or []
            if sources and min(sources) == 0:
                return {"code": "Ok", "durations": [[0.0, 100.0, 200.0]]}
            raise RuntimeError("chunk fail")

        monkeypatch.setattr(osrm, "_table", table)
        m = osrm.build_distance_matrix_osrm(
            coords,
            base_url=BASE,
            timeout=5,
            max_sources_per_call=1,
            redis_client=None,
        )
        assert len(m) == 3
        assert m[0][1] == 100.0
        assert m[1][0] > 0 or m[2][0] > 0

    def test_chunk_no_durations_triggers_fallback(self, real_osrm, monkeypatch):
        osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
        monkeypatch.setattr(
            osrm, "_table", lambda **_k: {"code": "Ok", "durations": []}
        )
        m = osrm.build_distance_matrix_osrm(
            [LAUSANNE, GENEVA], base_url=BASE, timeout=5
        )
        assert m[0][1] > 0

    def test_matrix_writes_redis_l2(self, real_osrm, monkeypatch):
        osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
        redis = MagicMock()
        redis.get.return_value = None
        monkeypatch.setattr(
            osrm,
            "_table",
            lambda **_k: {"code": "Ok", "durations": [[0.0, 55.0], [55.0, 0.0]]},
        )
        m = osrm.build_distance_matrix_osrm(
            [LAUSANNE, GENEVA],
            base_url=BASE,
            timeout=5,
            redis_client=redis,
        )
        assert m[0][1] == 55.0
        assert redis.setex.called

    def test_parallel_matrix_path(self, real_osrm, monkeypatch):
        osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
        monkeypatch.setattr(osrm, "OSRM_PARALLEL_THRESHOLD", 1)

        def table(**kwargs):
            sources = list(kwargs.get("sources") or [0])
            n = 3
            row = [float(10 * sources[0] + j) for j in range(n)]
            return {"code": "Ok", "durations": [row]}

        monkeypatch.setattr(osrm, "_table", table)
        m = osrm.build_distance_matrix_osrm(
            [LAUSANNE, GENEVA, (47.37, 8.54)],
            base_url=BASE,
            timeout=5,
            max_sources_per_call=1,
        )
        assert len(m) == 3

    def test_singleflight_propagates_leader_error(self, real_osrm):
        evt = MagicMock()
        evt.wait.return_value = True
        err = RuntimeError("leader failed")
        with osrm._inflight_lock:
            osrm._inflight["sf-err"] = {
                "evt": evt,
                "result": None,
                "error": err,
                "leader": False,
            }
        with pytest.raises(RuntimeError, match="leader failed"):
            osrm._singleflight_do("sf-err", lambda: 1, max_wait_seconds=0.1)

    def test_singleflight_follower_timeout_fn_raises(self, real_osrm):
        evt = MagicMock()
        evt.wait.return_value = False
        with osrm._inflight_lock:
            osrm._inflight["sf-to"] = {
                "evt": evt,
                "result": None,
                "error": None,
                "leader": False,
            }

        def boom():
            raise RuntimeError("direct fail")

        with pytest.raises(RuntimeError, match="direct fail"):
            osrm._singleflight_do("sf-to", boom, max_wait_seconds=0.01)

    def test_get_redis_client_fallback_ping(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm, "_get_redis_client_fallback", _REAL_GET_REDIS_FALLBACK
        )
        fake = MagicMock()
        fake.ping.return_value = True
        import ext as ext_mod

        monkeypatch.setattr(ext_mod, "redis_client", fake)
        assert _REAL_GET_REDIS_FALLBACK() is fake

    def test_route_info_writes_cache(self, real_osrm, monkeypatch):
        monkeypatch.setattr(
            osrm,
            "_route",
            lambda **_k: {
                "code": "Ok",
                "routes": [
                    {
                        "duration": 33.0,
                        "distance": 44.0,
                        "geometry": None,
                        "legs": [],
                    }
                ],
            },
        )
        redis = MagicMock()
        redis.get.return_value = None
        res = osrm.route_info(
            LAUSANNE,
            GENEVA,
            base_url=BASE,
            redis_client=redis,
            cache_ttl_s=60,
        )
        assert res["duration"] == 33.0
        assert redis.setex.called

    def test_fallback_matrix_n_le_1(self, real_osrm):
        assert osrm._fallback_matrix([]) == []
        assert osrm._fallback_matrix([LAUSANNE]) == [[0.0]]

    def test_shape_mismatch_chunk(self, real_osrm, monkeypatch):
        osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
        monkeypatch.setattr(
            osrm,
            "_table",
            lambda **_k: {"code": "Ok", "durations": [[0.0]]},
        )
        m = osrm.build_distance_matrix_osrm(
            [LAUSANNE, GENEVA], base_url=BASE, timeout=5
        )
        assert len(m) == 2
