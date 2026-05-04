"""Tests pour GET /api/v1/platform/status."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services.platform_runtime import (
    assert_runtime_payload_section_contract,
    build_platform_runtime_payload,
)
from services.platform_status_aggregator import (
    _build_metadata_block,
    _platform_setting,
    build_platform_status_payload,
    compute_overall_status,
)


@pytest.fixture(autouse=True)
def _clear_platform_env_for_tests(monkeypatch):
    """Évite qu'un export shell local ne fausse les tests (priorité env dans l'agrégateur)."""
    for k in (
        "PLATFORM_API_URL_PROD",
        "PLATFORM_API_URL_DEMO",
        "PLATFORM_LINK_GRAFANA",
        "PLATFORM_LINK_PROMETHEUS",
        "PLATFORM_LINK_ALERTMANAGER",
        "PLATFORM_STATUS_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(k, raising=False)


def test_compute_overall_status_demo_not_monitored():
    prod = {"monitored": True, "status": "ok"}
    demo = {"monitored": False, "status": "unknown"}
    assert compute_overall_status(prod, demo) == "ok"


def test_compute_overall_status_both_ok():
    prod = {"monitored": True, "status": "ok"}
    demo = {"monitored": True, "status": "ok"}
    assert compute_overall_status(prod, demo) == "ok"


def test_compute_overall_status_prod_unknown():
    prod = {"monitored": True, "status": "unknown"}
    demo = {"monitored": True, "status": "ok"}
    assert compute_overall_status(prod, demo) == "unknown"


def test_compute_overall_status_prod_not_monitored():
    """Prod non monitorée → impossible de conclure au niveau plateforme."""
    prod = {"monitored": False, "status": "unknown"}
    demo = {"monitored": True, "status": "ok"}
    assert compute_overall_status(prod, demo) == "unknown"


def test_compute_overall_status_prod_down():
    prod = {"monitored": True, "status": "down"}
    demo = {"monitored": True, "status": "ok"}
    assert compute_overall_status(prod, demo) == "down"


def test_compute_overall_status_demo_degraded():
    prod = {"monitored": True, "status": "ok"}
    demo = {"monitored": True, "status": "degraded"}
    assert compute_overall_status(prod, demo) == "degraded"


def test_build_platform_status_payload_not_monitored():
    cfg = MagicMock()
    cfg.PLATFORM_API_URL_PROD = None
    cfg.PLATFORM_API_URL_DEMO = None
    cfg.PLATFORM_STATUS_TIMEOUT_SECONDS = 2.5
    cfg.PLATFORM_LINK_GRAFANA = None
    cfg.PLATFORM_LINK_PROMETHEUS = None
    cfg.PLATFORM_LINK_ALERTMANAGER = None

    out = build_platform_status_payload(cfg)
    assert out["overall_status"] == "unknown"
    assert out["environments"]["prod"]["monitored"] is False
    assert out["environments"]["demo"]["monitored"] is False
    assert out["links"]["grafana"] is None
    assert out["metadata"]["status"] == "not_configured"
    assert out["deep_links"]["observability"]["grafana"] is None


def test_build_platform_runtime_payload_shape():
    out = build_platform_runtime_payload()
    assert "generated_at" in out
    assert "sections" in out
    assert_runtime_payload_section_contract(out)
    assert out["sections"]["process"]["status"] == "ok"
    assert "pid" in (out["sections"]["process"].get("data") or {})
    rs = out["sections"]["redis"]
    assert rs["status"] in ("ok", "unknown", "degraded")


@patch("redis.Redis")
def test_runtime_redis_section_ok(mock_redis_cls):
    mock_r = MagicMock()
    mock_r.ping.return_value = True
    mock_r.info.return_value = {
        "used_memory": 99_999,
        "used_memory_human": "97.66K",
        "connected_clients": 2,
        "uptime_in_seconds": 120,
        "evicted_keys": 0,
        "keyspace_hits": 5,
        "keyspace_misses": 1,
    }
    mock_redis_cls.from_url.return_value = mock_r
    out = build_platform_runtime_payload()
    assert_runtime_payload_section_contract(out)
    redis_sec = out["sections"]["redis"]
    assert redis_sec["status"] == "ok"
    assert redis_sec["data"]["ping_ok"] is True
    assert redis_sec["data"]["used_memory_bytes"] == 99_999


@patch("redis.Redis")
def test_runtime_redis_section_unreachable(mock_redis_cls):
    mock_redis_cls.from_url.side_effect = OSError("connection refused")
    out = build_platform_runtime_payload()
    assert_runtime_payload_section_contract(out)
    assert out["sections"]["redis"]["status"] == "unknown"
    assert out["sections"]["redis"]["reason"] == "redis_unreachable"


@patch("redis.Redis")
def test_runtime_redis_info_incomplete_after_ping(mock_redis_cls):
    """PING OK mais INFO sans champs mémoire → degraded + redis_info_parse_failed."""
    mock_r = MagicMock()
    mock_r.ping.return_value = True
    mock_r.info.return_value = {}
    mock_redis_cls.from_url.return_value = mock_r
    out = build_platform_runtime_payload()
    assert_runtime_payload_section_contract(out)
    assert out["sections"]["redis"]["status"] == "degraded"
    assert out["sections"]["redis"]["reason"] == "redis_info_parse_failed"


@patch("celery_app.celery.control.inspect")
@patch("redis.Redis")
def test_runtime_celery_section_ok(mock_redis_cls, mock_celery_inspect):
    mock_r = MagicMock()
    mock_r.ping.return_value = True
    mock_r.info.return_value = {
        "used_memory": 1000,
        "used_memory_human": "1.00K",
        "connected_clients": 1,
        "uptime_in_seconds": 60,
        "evicted_keys": 0,
        "keyspace_hits": 1,
        "keyspace_misses": 0,
    }
    mock_redis_cls.from_url.return_value = mock_r
    mock_insp = MagicMock()
    mock_insp.ping.return_value = {"celery@worker1": {"ok": "pong"}}
    mock_insp.stats.return_value = {"celery@worker1": {"pool": {"max-concurrency": 4}}}
    mock_celery_inspect.return_value = mock_insp
    out = build_platform_runtime_payload()
    assert_runtime_payload_section_contract(out)
    cel = out["sections"]["celery"]
    assert cel["status"] == "ok"
    assert cel["data"]["inspect_ok"] is True
    assert cel["data"]["workers_count"] == 1


@patch("celery_app.celery.control.inspect")
@patch("redis.Redis")
def test_runtime_celery_stats_empty_per_worker_degraded(
    mock_redis_cls, mock_celery_inspect
):
    """Ping OK mais stats vides par worker → degraded (pas de ok trop optimiste)."""
    mock_r = MagicMock()
    mock_r.ping.return_value = True
    mock_r.info.return_value = {
        "used_memory": 1000,
        "used_memory_human": "1.00K",
        "connected_clients": 1,
        "uptime_in_seconds": 60,
        "evicted_keys": 0,
        "keyspace_hits": 0,
        "keyspace_misses": 0,
    }
    mock_redis_cls.from_url.return_value = mock_r
    mock_insp = MagicMock()
    mock_insp.ping.return_value = {"celery@worker1": {"ok": "pong"}}
    mock_insp.stats.return_value = {"celery@worker1": {}}
    mock_celery_inspect.return_value = mock_insp
    out = build_platform_runtime_payload()
    assert out["sections"]["celery"]["status"] == "degraded"
    assert out["sections"]["celery"]["reason"] == "celery_partial_data"


@patch("celery_app.celery.control.inspect")
@patch("redis.Redis")
def test_runtime_celery_stats_worker_key_mismatch_degraded(
    mock_redis_cls, mock_celery_inspect
):
    """Stats non vides mais sans recoupement avec les workers du ping → degraded."""
    mock_r = MagicMock()
    mock_r.ping.return_value = True
    mock_r.info.return_value = {
        "used_memory": 1000,
        "used_memory_human": "1.00K",
        "connected_clients": 1,
        "uptime_in_seconds": 60,
        "evicted_keys": 0,
        "keyspace_hits": 0,
        "keyspace_misses": 0,
    }
    mock_redis_cls.from_url.return_value = mock_r
    mock_insp = MagicMock()
    mock_insp.ping.return_value = {"celery@a": {"ok": "pong"}}
    mock_insp.stats.return_value = {"celery@b": {"pool": {"max-concurrency": 4}}}
    mock_celery_inspect.return_value = mock_insp
    out = build_platform_runtime_payload()
    assert out["sections"]["celery"]["status"] == "degraded"
    assert out["sections"]["celery"]["reason"] == "celery_partial_data"


@patch("celery_app.celery.control.inspect")
@patch("redis.Redis")
def test_runtime_celery_ping_timeout_error_reason(mock_redis_cls, mock_celery_inspect):
    mock_r = MagicMock()
    mock_r.ping.return_value = True
    mock_r.info.return_value = {
        "used_memory": 1000,
        "used_memory_human": "1.00K",
        "connected_clients": 1,
        "uptime_in_seconds": 60,
        "evicted_keys": 0,
        "keyspace_hits": 0,
        "keyspace_misses": 0,
    }
    mock_redis_cls.from_url.return_value = mock_r
    mock_celery_inspect.side_effect = TimeoutError()
    out = build_platform_runtime_payload()
    assert out["sections"]["celery"]["reason"] == "celery_inspect_timeout"


@patch("celery_app.celery.control.inspect")
@patch("redis.Redis")
def test_runtime_celery_inspect_raises_other_sections_ok(
    mock_redis_cls, mock_celery_inspect
):
    """Échec Celery isolé : process + redis inchangés."""
    mock_r = MagicMock()
    mock_r.ping.return_value = True
    mock_r.info.return_value = {
        "used_memory": 1000,
        "used_memory_human": "1.00K",
        "connected_clients": 1,
        "uptime_in_seconds": 60,
        "evicted_keys": 0,
        "keyspace_hits": 0,
        "keyspace_misses": 0,
    }
    mock_redis_cls.from_url.return_value = mock_r
    mock_celery_inspect.side_effect = RuntimeError("inspect boom")
    out = build_platform_runtime_payload()
    assert_runtime_payload_section_contract(out)
    assert out["sections"]["process"]["status"] == "ok"
    assert out["sections"]["redis"]["status"] == "ok"
    assert out["sections"]["celery"]["status"] == "unknown"
    assert out["sections"]["celery"]["reason"] == "celery_inspect_failed"


def test_build_metadata_block_ok_from_env(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT_SHA", "deadbeef")
    monkeypatch.setenv("APP_VERSION", "1.2.3")
    cfg = MagicMock()
    meta = _build_metadata_block(cfg)
    assert meta["status"] == "ok"
    assert meta["data"]["git_commit"] == "deadbeef"
    assert meta["data"]["app_version"] == "1.2.3"


def test_platform_setting_env_overrides_config(monkeypatch):
    """Les variables PLATFORM_* dans l'environnement priment sur app.config."""
    monkeypatch.setenv("PLATFORM_API_URL_PROD", "https://from-env.example")
    cfg = MagicMock()
    cfg.PLATFORM_API_URL_PROD = "http://ignored-from-config"
    assert _platform_setting(cfg, "PLATFORM_API_URL_PROD") == "https://from-env.example"


@patch("services.platform_status_aggregator._fetch_json")
def test_build_platform_status_payload_monitored_ok(mock_fetch):
    def side_effect(base_url, path, timeout):
        if "/ready" in path:
            return (
                {"status": "ready", "checks": {"database": "ok", "redis": "ok"}},
                200,
                12.0,
                None,
            )
        if "websocket" in path:
            return ({"status": "ok"}, 200, 15.0, None)
        return None, None, 0.0, "bad path"

    mock_fetch.side_effect = side_effect

    cfg = MagicMock()
    cfg.PLATFORM_API_URL_PROD = "http://prod.example"
    cfg.PLATFORM_API_URL_DEMO = "http://demo.example"
    cfg.PLATFORM_STATUS_TIMEOUT_SECONDS = 2.5
    cfg.PLATFORM_LINK_GRAFANA = "https://grafana.example"
    cfg.PLATFORM_LINK_PROMETHEUS = None
    cfg.PLATFORM_LINK_ALERTMANAGER = None

    out = build_platform_status_payload(cfg)
    assert out["overall_status"] == "ok"
    assert out["global_status"] == "ok"
    assert out["environments"]["prod"]["status"] == "ok"
    assert out["environments"]["demo"]["status"] == "ok"
    assert out["environments"]["prod"]["name"] == "ATMR Production"
    rc = out["environments"]["prod"]["checks"]["ready"]
    assert rc["status"] == "ok"
    assert rc["criticality"] == "critical"
    assert rc["latency_ms"] is not None
    assert out["summary"]["total_checks"] == 8
    assert out["links"]["grafana"] == "https://grafana.example"


def _demo_health_ok(base_url, path, timeout):
    if "/ready" in path:
        return (
            {"status": "ready", "checks": {"database": "ok", "redis": "ok"}},
            200,
            12.0,
            None,
        )
    if "websocket" in path:
        return ({"status": "ok"}, 200, 15.0, None)
    return None, None, 0.0, "bad path"


@patch("services.platform_status_aggregator._fetch_json")
def test_build_platform_status_payload_prod_ready_timeout(mock_fetch):
    def side_effect(base_url, path, timeout):
        if "prod.example" in base_url:
            if "/ready" in path:
                return None, None, 100.0, "Read timed out"
            if "websocket" in path:
                return ({"status": "ok"}, 200, 15.0, None)
        if "demo.example" in base_url:
            return _demo_health_ok(base_url, path, timeout)
        return None, None, 0.0, "bad path"

    mock_fetch.side_effect = side_effect
    cfg = MagicMock()
    cfg.PLATFORM_API_URL_PROD = "http://prod.example"
    cfg.PLATFORM_API_URL_DEMO = "http://demo.example"
    cfg.PLATFORM_STATUS_TIMEOUT_SECONDS = 2.5
    cfg.PLATFORM_LINK_GRAFANA = None
    cfg.PLATFORM_LINK_PROMETHEUS = None
    cfg.PLATFORM_LINK_ALERTMANAGER = None

    out = build_platform_status_payload(cfg)
    assert out["environments"]["prod"]["status"] == "unknown"
    assert out["overall_status"] == "unknown"
    assert out["global_status"] == "unknown"
    assert "summary" in out


@patch("services.platform_status_aggregator._fetch_json")
def test_build_platform_status_payload_prod_websocket_timeout(mock_fetch):
    def side_effect(base_url, path, timeout):
        if "prod.example" in base_url:
            if "/ready" in path:
                return (
                    {"status": "ready", "checks": {"database": "ok", "redis": "ok"}},
                    200,
                    12.0,
                    None,
                )
            if "websocket" in path:
                return None, None, 100.0, "Read timed out"
        if "demo.example" in base_url:
            return _demo_health_ok(base_url, path, timeout)
        return None, None, 0.0, "bad path"

    mock_fetch.side_effect = side_effect
    cfg = MagicMock()
    cfg.PLATFORM_API_URL_PROD = "http://prod.example"
    cfg.PLATFORM_API_URL_DEMO = "http://demo.example"
    cfg.PLATFORM_STATUS_TIMEOUT_SECONDS = 2.5
    cfg.PLATFORM_LINK_GRAFANA = None
    cfg.PLATFORM_LINK_PROMETHEUS = None
    cfg.PLATFORM_LINK_ALERTMANAGER = None

    out = build_platform_status_payload(cfg)
    # WebSocket seul en unknown (high) : environnement prod dégradé (plan §3.3)
    assert out["environments"]["prod"]["status"] == "degraded"
    assert out["overall_status"] == "degraded"
    assert out["global_status"] == "degraded"


@patch("services.platform_status_aggregator._fetch_json")
def test_build_platform_status_payload_prod_ready_and_websocket_timeout(mock_fetch):
    def side_effect(base_url, path, timeout):
        if "prod.example" in base_url:
            if "/ready" in path:
                return None, None, 100.0, "Read timed out"
            if "websocket" in path:
                return None, None, 100.0, "Read timed out"
        if "demo.example" in base_url:
            return _demo_health_ok(base_url, path, timeout)
        return None, None, 0.0, "bad path"

    mock_fetch.side_effect = side_effect
    cfg = MagicMock()
    cfg.PLATFORM_API_URL_PROD = "http://prod.example"
    cfg.PLATFORM_API_URL_DEMO = "http://demo.example"
    cfg.PLATFORM_STATUS_TIMEOUT_SECONDS = 2.5
    cfg.PLATFORM_LINK_GRAFANA = None
    cfg.PLATFORM_LINK_PROMETHEUS = None
    cfg.PLATFORM_LINK_ALERTMANAGER = None

    out = build_platform_status_payload(cfg)
    assert out["environments"]["prod"]["status"] == "unknown"
    assert out["overall_status"] == "unknown"
