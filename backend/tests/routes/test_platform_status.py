"""Tests pour GET /api/v1/platform/status."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from services.platform_status_aggregator import (
    build_platform_status_payload,
    compute_overall_status,
)


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


def test_compute_overall_status_prod_unavailable():
    prod = {"monitored": True, "status": "unavailable"}
    demo = {"monitored": True, "status": "ok"}
    assert compute_overall_status(prod, demo) == "unavailable"


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
    assert out["environments"]["prod"]["status"] == "ok"
    assert out["environments"]["demo"]["status"] == "ok"
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
    assert out["environments"]["prod"]["status"] == "unknown"
    assert out["overall_status"] == "unknown"


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
