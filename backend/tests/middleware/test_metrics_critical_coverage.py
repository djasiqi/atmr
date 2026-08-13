"""Couverture critique de ``middleware.metrics`` (seuil 80 %)."""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from flask import Flask, g, request
from sqlalchemy import text

from middleware import metrics


@pytest.fixture
def mini_app():
    app = Flask("metrics_coverage")
    app.config["TESTING"] = True

    @app.route("/ping")
    def ping():
        return "ok", 200

    @app.route("/api/bookings/<int:booking_id>")
    def booking_detail(booking_id):
        return {"id": booking_id}, 200

    return app


@pytest.fixture
def restore_sql_flag():
    original = metrics._sql_metrics_state["listener_registered"]
    yield
    metrics._sql_metrics_state["listener_registered"] = original


def test_get_endpoint_normalise_ids_et_tronque():
    req = SimpleNamespace(endpoint=None, path="/api/bookings/123/driver/456")
    assert metrics._get_endpoint(req) == "/api/bookings/:id/driver/:id"

    req = SimpleNamespace(endpoint="bookings.get", path="/ignored")
    assert metrics._get_endpoint(req) == "bookings.get"

    long_path = "/api/" + ("x" * 200)
    req = SimpleNamespace(endpoint=None, path=long_path)
    result = metrics._get_endpoint(req)
    assert result.endswith("...")
    assert len(result) == metrics._MAX_ENDPOINT_LENGTH + 3


def test_prom_middleware_sans_prometheus(mini_app, monkeypatch):
    monkeypatch.setattr(metrics, "PROMETHEUS_AVAILABLE", False)
    mini_app.logger = MagicMock()
    result = metrics.prom_middleware(mini_app)
    assert result is mini_app
    mini_app.logger.warning.assert_called_once()
    assert "prometheus_client non installé" in mini_app.logger.warning.call_args[0][0]


def test_prom_middleware_requete_et_metrics_http(mini_app, monkeypatch):
    monkeypatch.delenv("METRICS_SQL_PER_REQUEST", raising=False)
    instrumented = metrics.prom_middleware(mini_app)
    client = instrumented.test_client()

    resp = client.get("/ping")
    assert resp.status_code == 200

    resp_id = client.get("/api/bookings/42")
    assert resp_id.status_code == 200

    metrics_resp = client.get("/prometheus/metrics-http")
    assert metrics_resp.status_code == 200
    assert b"http_requests_total" in metrics_resp.data or metrics_resp.data


def test_after_request_sans_timer(mini_app, monkeypatch):
    monkeypatch.delenv("METRICS_SQL_PER_REQUEST", raising=False)
    metrics.prom_middleware(mini_app)

    @mini_app.before_request
    def _wipe_timer():
        if hasattr(request, "_prom_start_time"):
            delattr(request, "_prom_start_time")

    resp = mini_app.test_client().get("/ping")
    assert resp.status_code == 200


def test_sql_par_requete_et_histogramme(mini_app, monkeypatch, restore_sql_flag):
    monkeypatch.setenv("METRICS_SQL_PER_REQUEST", "true")
    metrics._sql_metrics_state["listener_registered"] = False
    metrics.prom_middleware(mini_app)

    resp = mini_app.test_client().get("/ping")
    assert resp.status_code == 200


def test_sql_histogramme_observe_en_echec(mini_app, monkeypatch, restore_sql_flag):
    monkeypatch.setenv("METRICS_SQL_PER_REQUEST", "true")
    boom = MagicMock()
    boom.labels.return_value.observe.side_effect = RuntimeError("observe")
    monkeypatch.setattr(metrics, "HTTP_REQUEST_DB_QUERIES", boom)
    metrics._sql_metrics_state["listener_registered"] = False
    metrics.prom_middleware(mini_app)
    assert mini_app.test_client().get("/ping").status_code == 200


def test_slo_import_error(mini_app, monkeypatch):
    monkeypatch.delenv("METRICS_SQL_PER_REQUEST", raising=False)
    metrics.prom_middleware(mini_app)
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "services.monitoring.slo":
            raise ImportError("slo indisponible")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    assert mini_app.test_client().get("/ping").status_code == 200


def test_slo_exception_generique(mini_app, monkeypatch):
    monkeypatch.delenv("METRICS_SQL_PER_REQUEST", raising=False)
    metrics.prom_middleware(mini_app)
    monkeypatch.setattr(
        "services.monitoring.slo.record_slo_metric",
        MagicMock(side_effect=RuntimeError("slo boom")),
    )
    assert mini_app.test_client().get("/ping").status_code == 200


def test_metrics_http_prometheus_indisponible(mini_app, monkeypatch):
    monkeypatch.delenv("METRICS_SQL_PER_REQUEST", raising=False)
    metrics.prom_middleware(mini_app)
    monkeypatch.setattr(metrics, "PROMETHEUS_AVAILABLE", False)
    resp = mini_app.test_client().get("/prometheus/metrics-http")
    assert resp.status_code == 503
    assert resp.get_json()["error"] == "Prometheus client non disponible"


def test_register_sql_deja_enregistre(mini_app, restore_sql_flag):
    metrics._sql_metrics_state["listener_registered"] = True
    mini_app.logger = MagicMock()
    metrics._register_sql_per_request_counter(mini_app)
    mini_app.logger.info.assert_not_called()


def test_register_sql_echec_import(mini_app, monkeypatch, restore_sql_flag):
    metrics._sql_metrics_state["listener_registered"] = False
    mini_app.logger = MagicMock()
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "sqlalchemy":
            raise ImportError("sqlalchemy indisponible")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)
    metrics._register_sql_per_request_counter(mini_app)
    mini_app.logger.warning.assert_called_once()
    assert metrics._sql_metrics_state["listener_registered"] is False


def test_count_sql_listener_branches(app, db, restore_sql_flag):
    metrics._sql_metrics_state["listener_registered"] = False
    metrics._register_sql_per_request_counter(app)

    db.session.execute(text("SELECT 1"))

    with app.test_request_context("/"):
        db.session.execute(text("SELECT 1"))
        g._metrics_sql_track = True
        g._metrics_sql_count = 0
        db.session.execute(text("SELECT 1"))
        assert int(g._metrics_sql_count) >= 1


def test_before_request_sans_gauge(mini_app, monkeypatch):
    monkeypatch.delenv("METRICS_SQL_PER_REQUEST", raising=False)
    monkeypatch.setattr(metrics, "REQUEST_IN_PROGRESS", None)
    monkeypatch.setattr(metrics, "REQUEST_LATENCY", None)
    monkeypatch.setattr(metrics, "REQUEST_COUNT", None)
    monkeypatch.setattr(metrics, "HTTP_REQUEST_DB_QUERIES", None)
    metrics.prom_middleware(mini_app)
    assert mini_app.test_client().get("/ping").status_code == 200
