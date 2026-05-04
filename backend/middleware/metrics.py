"""✅ Middleware Prometheus pour métriques HTTP (latence, compteurs).

Instrumente toutes les requêtes HTTP pour exposer:
- Latence p50/p95/p99 via histogram
- Compteurs de requêtes par méthode/endpoint/status
- Optionnel : nombre de requêtes SQL par requête HTTP (METRICS_SQL_PER_REQUEST=true)
"""

import os
import time
from contextlib import suppress
from typing import TYPE_CHECKING

from flask import Flask, g, has_request_context, request

if TYPE_CHECKING:
    from flask import Response

# Import optionnel prometheus_client (peut ne pas être installé en dev)
try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Fallback si prometheus_client n'est pas installé
    generate_latest = None
    Counter = None
    Histogram = None
    Gauge = None


# Métriques Prometheus (créées uniquement si prometheus_client disponible)
if (
    PROMETHEUS_AVAILABLE
    and Histogram is not None
    and Counter is not None
    and Gauge is not None
):
    REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency en secondes",
        ["method", "endpoint", "status"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    REQUEST_COUNT = Counter(
        "http_requests_total",
        "Nombre total de requêtes HTTP",
        ["method", "endpoint", "status"],
    )

    REQUEST_IN_PROGRESS = Gauge(
        "http_requests_in_progress",
        "Nombre de requêtes en cours",
        ["method", "endpoint"],
    )

    HTTP_REQUEST_DB_QUERIES = Histogram(
        "http_request_db_queries",
        "Nombre de requêtes SQL exécutées pendant une requête HTTP",
        ["method", "endpoint"],
        buckets=(0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 500),
    )
else:
    REQUEST_LATENCY = None
    REQUEST_COUNT = None
    REQUEST_IN_PROGRESS = None
    HTTP_REQUEST_DB_QUERIES = None


_sql_metrics_state = {"listener_registered": False}


def _register_sql_per_request_counter(app: Flask) -> None:
    """Compte les SQLAlchemy cursor executes par requête HTTP (best-effort)."""
    if _sql_metrics_state["listener_registered"]:
        return
    try:
        from sqlalchemy import event as sqlalchemy_event
        from sqlalchemy.engine import Engine

        @sqlalchemy_event.listens_for(Engine, "before_cursor_execute")
        def _count_sql(  # pyright: ignore[reportUnusedFunction]
            _conn, _cursor, _statement, _parameters, _context, _executemany
        ):
            if not has_request_context():
                return
            if not getattr(g, "_metrics_sql_track", False):
                return
            with suppress(Exception):
                g._metrics_sql_count = int(getattr(g, "_metrics_sql_count", 0)) + 1

        _sql_metrics_state["listener_registered"] = True
        app.logger.info("[Prometheus] Compteur SQL par requête HTTP activé")
    except Exception as e:
        app.logger.warning("[Prometheus] SQL per request non disponible: %s", e)


def prom_middleware(app: Flask) -> Flask:
    """Ajoute le middleware Prometheus à l'application Flask.

    Args:
        app: Instance Flask

    Returns:
        App Flask avec middleware ajouté
    """
    if not PROMETHEUS_AVAILABLE:
        warning_msg = (
            "[Prometheus] prometheus_client non installé - "
            "métriques HTTP désactivées. "
            "Installer avec: pip install prometheus-client"
        )
        app.logger.warning(warning_msg)
        return app

    app.logger.info("[Prometheus] Middleware métriques HTTP activé")

    if os.getenv("METRICS_SQL_PER_REQUEST", "false").lower() in ("1", "true", "yes"):
        _register_sql_per_request_counter(app)

    @app.before_request
    def _start_timer():  # pyright: ignore[reportUnusedFunction]
        """Marque le début de la requête."""
        request._prom_start_time = time.time()
        if os.getenv("METRICS_SQL_PER_REQUEST", "false").lower() in (
            "1",
            "true",
            "yes",
        ):
            g._metrics_sql_track = True
            g._metrics_sql_count = 0

        # Incrémenter compteur requêtes en cours
        if REQUEST_IN_PROGRESS:
            endpoint = _get_endpoint(request)
            REQUEST_IN_PROGRESS.labels(method=request.method, endpoint=endpoint).inc()

    @app.after_request
    def _record_metrics(  # pyright: ignore[reportUnusedFunction]
        resp: "Response",
    ) -> "Response":
        """Enregistre les métriques après la requête."""
        if not hasattr(request, "_prom_start_time"):
            return resp

        # Calculer durée
        duration = time.time() - request._prom_start_time

        # Normaliser endpoint (enlever IDs dynamiques)
        endpoint = _get_endpoint(request)

        # Décrémenter requêtes en cours
        if REQUEST_IN_PROGRESS:
            REQUEST_IN_PROGRESS.labels(method=request.method, endpoint=endpoint).dec()

        # Métriques Prometheus standards
        if REQUEST_LATENCY:
            REQUEST_LATENCY.labels(
                method=request.method, endpoint=endpoint, status=resp.status_code
            ).observe(duration)

        if REQUEST_COUNT:
            REQUEST_COUNT.labels(
                method=request.method, endpoint=endpoint, status=resp.status_code
            ).inc()

        if (
            HTTP_REQUEST_DB_QUERIES
            and getattr(g, "_metrics_sql_track", False)
            and hasattr(g, "_metrics_sql_count")
        ):
            with suppress(Exception):
                HTTP_REQUEST_DB_QUERIES.labels(
                    method=request.method, endpoint=endpoint
                ).observe(float(g._metrics_sql_count))

        # ✅ SLO: Enregistrer métriques SLO pour routes critiques
        try:
            from services.monitoring.slo import record_slo_metric

            record_slo_metric(
                endpoint=endpoint,
                duration_seconds=duration,
                status_code=resp.status_code,
                method=request.method,
            )
        except ImportError:
            # api_slo.py peut ne pas être disponible en dev
            pass
        except Exception:
            # Ne pas faire échouer la requête si SLO tracking échoue
            pass

        return resp

    # Endpoint pour exporter métriques Prometheus
    @app.route("/prometheus/metrics-http")
    def metrics_http():  # pyright: ignore[reportUnusedFunction]
        """Exporte les métriques HTTP au format Prometheus."""
        if not PROMETHEUS_AVAILABLE or generate_latest is None:
            from flask import jsonify

            return jsonify(
                {
                    "error": "Prometheus client non disponible",
                    "message": "Installer avec: pip install prometheus-client",
                }
            ), 503

        from flask import Response

        return Response(
            generate_latest(), mimetype="text/plain; version=0.0.4; charset=utf-8"
        )

    return app


_MAX_ENDPOINT_LENGTH = 100


def _get_endpoint(request) -> str:
    """Normalise l'endpoint pour éviter explosion de labels.

    Remplace les IDs numériques par :id pour regrouper les routes.
    Ex: /api/bookings/123 → /api/bookings/:id

    Args:
        request: Flask request object

    Returns:
        Endpoint normalisé
    """
    endpoint = request.endpoint or request.path

    # Si c'est un path avec des IDs, normaliser
    import re

    # Pattern: /api/resource/123 ou /api/resource/123/subresource/456
    endpoint = re.sub(r"/\d+(?=/|$)", "/:id", endpoint)

    # Limiter longueur (éviter labels trop longs)
    if len(endpoint) > _MAX_ENDPOINT_LENGTH:
        endpoint = endpoint[:_MAX_ENDPOINT_LENGTH] + "..."

    return endpoint
