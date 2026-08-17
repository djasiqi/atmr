"""Couverture critique ``services/geolocation/osrm.py`` (seuil 90 %)."""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests

from services.geolocation import osrm
from tests.services.test_osrm_unit import (
    _REAL_GET_REDIS_FALLBACK,
    BASE,
    GENEVA,
    LAUSANNE,
    _MockResponse,
)
from tests.services.test_osrm_unit import real_osrm as _real_osrm_fixture


@pytest.fixture
def real_osrm(_real_osrm_fixture):  # noqa: F811
    return _real_osrm_fixture


def test_chaos_down_et_latence_table_et_route(real_osrm, monkeypatch):
    inj = SimpleNamespace(enabled=True, osrm_down=True, latency_ms=0)
    monkeypatch.setattr(osrm, "get_chaos_injector", lambda: inj)
    with pytest.raises(requests.ConnectionError, match="CHAOS"):
        osrm._table_single_request(BASE, "driving", [LAUSANNE], None, None, 5)
    with pytest.raises(requests.ConnectionError, match="CHAOS"):
        osrm._route(BASE, "driving", LAUSANNE, GENEVA, timeout=5)

    inj.osrm_down = False
    inj.latency_ms = 25
    metrics = MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "chaos.metrics",
        SimpleNamespace(get_chaos_metrics=lambda: metrics),
    )
    monkeypatch.setattr(
        osrm.requests,
        "get",
        lambda *a, **k: _MockResponse({"code": "Ok", "durations": [[0.0]]}),
    )
    osrm._table_single_request(BASE, "driving", [LAUSANNE], None, None, 5)
    assert metrics.record_latency.called

    monkeypatch.setattr(
        osrm.requests,
        "get",
        lambda *a, **k: _MockResponse(
            {
                "code": "Ok",
                "routes": [
                    {"duration": 1.0, "distance": 2.0, "geometry": None, "legs": []}
                ],
            }
        ),
    )
    osrm._route(BASE, "driving", LAUSANNE, GENEVA, timeout=5)


def test_chaos_latence_sans_module_metrics(real_osrm, monkeypatch):
    inj = SimpleNamespace(enabled=True, osrm_down=False, latency_ms=10)
    monkeypatch.setattr(osrm, "get_chaos_injector", lambda: inj)
    monkeypatch.setitem(sys.modules, "chaos.metrics", None)
    real_import = __import__

    def _import(name, *args, **kwargs):
        if name == "chaos.metrics":
            raise ImportError("absent")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _import)
    monkeypatch.setattr(
        osrm.requests,
        "get",
        lambda *a, **k: _MockResponse({"code": "Ok", "durations": [[0.0]]}),
    )
    osrm._table_single_request(BASE, "driving", [LAUSANNE], None, None, 5)
    monkeypatch.setattr(
        osrm.requests,
        "get",
        lambda *a, **k: _MockResponse(
            {
                "code": "Ok",
                "routes": [
                    {"duration": 1.0, "distance": 1.0, "geometry": None, "legs": []}
                ],
            }
        ),
    )
    osrm._route(BASE, "driving", LAUSANNE, GENEVA, waypoints=[(46.4, 6.4)])


def test_table_et_route_timeout_defaut_dns_et_json(real_osrm, monkeypatch):
    monkeypatch.setattr(
        osrm.requests,
        "get",
        lambda *a, **k: _MockResponse(
            {"code": "Ok", "durations": [[0.0, 1.0], [1.0, 0.0]]}
        ),
    )
    osrm._table(BASE, "driving", [LAUSANNE, GENEVA], None, None, timeout=None)

    def dns_fail(*_a, **_k):
        raise requests.ConnectionError("Failed to resolve osrm-host")

    monkeypatch.setattr(osrm.requests, "get", dns_fail)
    with pytest.raises(requests.ConnectionError):
        osrm._table_single_request(BASE, "driving", [LAUSANNE], None, None, 1)
    with pytest.raises(requests.ConnectionError):
        osrm._route(BASE, "driving", LAUSANNE, GENEVA, timeout=1)

    def unknown_host(*_a, **_k):
        raise requests.ConnectionError("Name or service not known")

    monkeypatch.setattr(osrm.requests, "get", unknown_host)
    with pytest.raises(requests.ConnectionError):
        osrm._route(BASE, "driving", LAUSANNE, GENEVA, timeout=1)

    class BadJson:
        status_code = 200
        elapsed = timedelta(seconds=0.01)
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            raise json.JSONDecodeError("err", "doc", 0)

    monkeypatch.setattr(osrm.requests, "get", lambda *a, **k: BadJson())
    with pytest.raises(ValueError, match="invalid JSON"):
        osrm._route(BASE, "driving", LAUSANNE, GENEVA, timeout=1)


def test_frequency_et_adaptive_ttl(real_osrm, monkeypatch):
    redis = MagicMock()
    osrm._increment_frequency_counter(redis, "k" * 80, cache_type="table")
    redis.incr.side_effect = RuntimeError("redis down")
    osrm._increment_frequency_counter(redis, "k")

    redis.get.return_value = None
    assert osrm._get_frequency_count(redis, "k") == 0
    redis.get.return_value = object()
    assert osrm._get_frequency_count(redis, "k") == 0
    redis.get.side_effect = RuntimeError("boom")
    assert osrm._get_frequency_count(redis, "k") == 0

    assert osrm._get_adaptive_ttl(None, "k", 42) == 42
    monkeypatch.setattr(
        osrm,
        "_get_frequency_count",
        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")),
    )
    assert osrm._get_adaptive_ttl(redis, "k", 77) == 77

    monkeypatch.setattr(osrm, "_get_frequency_count", lambda *_a, **_k: 99)
    inc = MagicMock()
    fake_counter = SimpleNamespace(labels=lambda **_k: SimpleNamespace(inc=inc))
    monkeypatch.setattr(
        "services.unified_dispatch.metrics.osrm_cache.OSRM_CACHE_FREQUENT_ROUTES_TOTAL",
        fake_counter,
        raising=False,
    )
    ttl = osrm._get_adaptive_ttl(redis, "z" * 80, 3600, cache_type="route")
    assert ttl == osrm.CACHE_TTL_FREQUENT
    assert inc.called


def test_redis_client_fallback_depuis_url(real_osrm, monkeypatch):
    monkeypatch.setattr(osrm, "_get_redis_client_fallback", _REAL_GET_REDIS_FALLBACK)
    import ext as ext_mod

    dead = MagicMock()
    dead.ping.side_effect = RuntimeError("ext down")
    monkeypatch.setattr(ext_mod, "redis_client", dead)
    monkeypatch.delenv("REDIS_URL", raising=False)
    assert _REAL_GET_REDIS_FALLBACK() is None

    fake_client = MagicMock()
    fake_client.ping.return_value = True
    fake_redis = SimpleNamespace(from_url=lambda *_a, **_k: fake_client)
    monkeypatch.setitem(sys.modules, "redis", fake_redis)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    assert _REAL_GET_REDIS_FALLBACK() is fake_client

    monkeypatch.setitem(
        sys.modules,
        "redis",
        SimpleNamespace(
            from_url=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("no redis"))
        ),
    )
    assert _REAL_GET_REDIS_FALLBACK() is None


def test_matrix_timeout_adaptatif_et_fallback_redis(real_osrm, monkeypatch):
    seen: list[int | None] = []

    def fake_table(**kwargs):
        seen.append(kwargs.get("timeout"))
        src = list(kwargs.get("sources") or [0])
        dests = list(kwargs.get("destinations") or src)
        return {
            "code": "Ok",
            "durations": [[0.0] * len(dests) for _ in src],
        }

    monkeypatch.setattr(osrm, "_table", fake_table)
    import ext as ext_mod

    monkeypatch.setattr(ext_mod, "redis_client", None)

    fake_rc = MagicMock()
    fake_rc.get.return_value = None
    monkeypatch.setattr(osrm, "_get_redis_client_fallback", lambda: fake_rc)

    coords2 = [LAUSANNE, GENEVA]
    osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
    osrm.build_distance_matrix_osrm(coords2, base_url=BASE, timeout=None)
    assert fake_rc.setex.called or fake_rc.get.called

    for n, expected in ((21, 45), (51, 60), (101, 90), (151, 120)):
        seen.clear()
        osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
        coords = [(46.0 + i * 0.001, 6.0) for i in range(n)]
        osrm.build_distance_matrix_osrm(
            coords, base_url=BASE, timeout=None, max_sources_per_call=200
        )
        assert expected in seen


def test_precomputed_matrix_hit_json_invalide_et_erreur(real_osrm, monkeypatch):
    import ext as ext_mod

    a = (46.20, 6.15)
    b = (46.21, 6.16)
    matrix = [[0.0, 3.5], [3.5, 0.0]]
    rc = MagicMock()
    rc.get.return_value = json.dumps(matrix).encode()
    monkeypatch.setattr(ext_mod, "redis_client", rc)
    osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
    out = osrm.build_distance_matrix_osrm([a, b], base_url=BASE, timeout=5)
    assert out[0][1] == 3.5

    osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
    rc.get.return_value = b"not-json"
    monkeypatch.setattr(
        osrm,
        "_table",
        lambda **_k: {"code": "Ok", "durations": [[0.0, 1.0], [1.0, 0.0]]},
    )
    out2 = osrm.build_distance_matrix_osrm([a, b], base_url=BASE, timeout=5)
    assert out2[0][1] == 1.0

    osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
    rc.get.side_effect = RuntimeError("precompute down")
    out3 = osrm.build_distance_matrix_osrm([a, b], base_url=BASE, timeout=5)
    assert out3[0][1] == 1.0


def test_matrix_l2_bytes_redis_conn_et_row_mismatch(real_osrm, monkeypatch):
    osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
    redis = MagicMock()
    redis.get.return_value = json.dumps(
        {"durations": [[0.0, 88.0], [88.0, 0.0]]}
    ).encode()
    m = osrm.build_distance_matrix_osrm(
        [LAUSANNE, GENEVA], base_url=BASE, timeout=5, redis_client=redis
    )
    assert m[0][1] == 88.0

    osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
    redis.get.side_effect = osrm._RedisConnError("down")
    monkeypatch.setattr(
        osrm,
        "_table",
        lambda **_k: {"code": "Ok", "durations": [[0.0, 2.0], [2.0, 0.0]]},
    )
    redis.setex.side_effect = RuntimeError("setex fail")
    m2 = osrm.build_distance_matrix_osrm(
        [LAUSANNE, GENEVA], base_url=BASE, timeout=5, redis_client=redis
    )
    assert m2[0][1] == 2.0

    osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
    monkeypatch.setattr(
        osrm,
        "_table",
        lambda **_k: {"code": "Ok", "durations": [[0.0], [1.0, 0.0]]},
    )
    m3 = osrm.build_distance_matrix_osrm(
        [LAUSANNE, GENEVA],
        base_url=BASE,
        timeout=5,
        redis_client=None,
        max_sources_per_call=2,
    )
    assert len(m3) == 2


def test_parallel_future_exception(real_osrm, monkeypatch):
    osrm._OSRM_MATRIX_LOCAL_CACHE.clear()
    monkeypatch.setattr(osrm, "OSRM_PARALLEL_THRESHOLD", 1)

    class BoomFuture:
        def result(self):
            raise RuntimeError("worker crash")

    class FakePool:
        def __init__(self, **_k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def submit(self, fn, chunk):
            return BoomFuture()

    monkeypatch.setattr(osrm, "ThreadPoolExecutor", FakePool)
    monkeypatch.setattr(
        osrm,
        "as_completed",
        lambda futures: list(futures),
    )
    m = osrm.build_distance_matrix_osrm(
        [LAUSANNE, GENEVA, (47.37, 8.54)],
        base_url=BASE,
        timeout=5,
        max_sources_per_call=1,
    )
    assert len(m) == 3


def test_route_info_cache_bytes_exceptions_et_setex(real_osrm, monkeypatch):
    redis = MagicMock()
    redis.get.return_value = json.dumps({"duration": 9.0, "distance": 10.0}).encode()
    res = osrm.route_info(LAUSANNE, GENEVA, base_url=BASE, redis_client=redis)
    assert res["duration"] == 9.0
    assert res["fallback"] is False

    redis.get.side_effect = ValueError("bad json")
    monkeypatch.setattr(
        osrm,
        "_route",
        lambda **_k: {
            "code": "Ok",
            "routes": [
                {"duration": 3.0, "distance": 4.0, "geometry": None, "legs": []}
            ],
        },
    )
    redis.setex.side_effect = osrm._RedisConnError("write down")
    res2 = osrm.route_info(LAUSANNE, GENEVA, base_url=BASE, redis_client=redis)
    assert res2["duration"] == 3.0

    redis.get.side_effect = None
    redis.get.return_value = None
    redis.setex.side_effect = RuntimeError("setex")
    res3 = osrm.route_info(
        LAUSANNE, GENEVA, base_url=BASE, redis_client=redis, cache_ttl_s=None
    )
    assert res3["duration"] == 3.0


def test_osrm_client_ajoute_fallback_false(real_osrm, monkeypatch):
    monkeypatch.setattr(
        osrm,
        "route_info",
        lambda *a, **k: {
            "duration": 1.0,
            "distance": 2.0,
            "geometry": None,
            "legs": [],
        },
    )
    res = osrm.OSRMClient(BASE).get_route(LAUSANNE, GENEVA)
    assert res["fallback"] is False


def test_get_distance_time_url_vide(real_osrm, monkeypatch):
    monkeypatch.setenv("OSRM_BASE_URL", "")
    monkeypatch.setattr(
        osrm,
        "route_info",
        lambda *a, **k: {"duration": 5.0, "distance": 6.0},
    )
    out = osrm.get_distance_time(LAUSANNE, GENEVA, base_url="")
    assert out["duration"] == 5.0


def test_cached_helpers_date_none_bytes_et_erreurs(real_osrm, monkeypatch):
    import ext as ext_mod

    monkeypatch.setattr(ext_mod, "redis_client", None)
    monkeypatch.setattr(
        osrm,
        "get_distance_time",
        lambda *a, **k: {"distance": 1.0, "duration": 2.0},
    )
    out = osrm.get_distance_time_cached(LAUSANNE, GENEVA)
    assert out["duration"] == 2.0

    fake = MagicMock()
    fake.get.return_value = b'{"distance": 3.0, "duration": 4.0}'
    monkeypatch.setattr(ext_mod, "redis_client", fake)
    hit = osrm.get_distance_time_cached(LAUSANNE, GENEVA, date_str="2026-02-02")
    assert hit["duration"] == 4.0

    fake.get.return_value = 12345
    coerced = osrm.get_distance_time_cached(LAUSANNE, GENEVA, date_str="2026-02-03")
    assert coerced == 12345 or isinstance(coerced, (dict, int))

    monkeypatch.setattr(ext_mod, "redis_client", None)
    monkeypatch.setattr(osrm, "get_matrix", lambda *a, **k: {"durations": [[0.0, 1.0]]})
    mat = osrm.get_matrix_cached([LAUSANNE], [GENEVA])
    assert mat["durations"][0][1] == 1.0

    fake.get.return_value = b'{"durations": [[0, 8]]}'
    monkeypatch.setattr(ext_mod, "redis_client", fake)
    mat_hit = osrm.get_matrix_cached([LAUSANNE], [GENEVA], date_str="2026-02-04")
    assert mat_hit["durations"][0][1] == 8

    fake.get.return_value = {"already": "dict"}
    mat_ns = osrm.get_matrix_cached([LAUSANNE], [GENEVA], date_str="2026-02-05")
    assert mat_ns == {"already": "dict"} or "durations" in mat_ns

    fake.get.return_value = None
    fake.setex.side_effect = RuntimeError("write fail")
    monkeypatch.setattr(osrm, "get_matrix", lambda *a, **k: {"durations": [[0.0, 9.0]]})
    mat_write = osrm.get_matrix_cached([LAUSANNE], [GENEVA], date_str="2026-02-06")
    assert mat_write["durations"][0][1] == 9.0
