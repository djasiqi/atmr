"""✅ Middleware Prometheus pour métriques HTTP (latence, compteurs).

Instrumente toutes les requêtes HTTP pour exposer:
- Latence p50/p95/p99 via histogram
- Compteurs de requêtes par méthode/endpoint/status
"""

import time
from typing import TYPE_CHECKING

from flask import Flask, request

if TYPE_CHECKING:
    from flask import Response

# Import optionnel prometheus_client (peut ne pas être installé en dev)
try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest

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
else:
    REQUEST_LATENCY = None
    REQUEST_COUNT = None
    REQUEST_IN_PROGRESS = None


def prom_middleware(app: Flask) -> Flask:
    """Ajoute le middleware Prometheus à l'application Flask.

    Args:
        app: Instance Flask

    Returns:
        App Flask avec middleware ajouté
    """
    if not PROMETHEUS_AVAILABLE:
        app.logger.warning(
            "[Prometheus] prometheus_client non installé - métriques HTTP désactivées. Installer avec: pip install prometheus-client"
        )
        return app

    app.logger.info("[Prometheus] Middleware métriques HTTP activé")

    @app.before_request
    def _start_timer():  # pyright: ignore[reportUnusedFunction]
        """Marque le début de la requête."""
        request._prom_start_time = time.time()

        # Incrémenter compteur requêtes en cours
        if REQUEST_IN_PROGRESS:
            endpoint = _get_endpoint(request)
            REQUEST_IN_PROGRESS.labels(method=request.method, endpoint=endpoint).inc()

    @app.after_request
    def _record_metrics(resp: "Response") -> "Response":  # pyright: ignore[reportUnusedFunction]
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

        # ✅ SLO: Enregistrer métriques SLO pour routes critiques
        try:
            from services.api_slo import record_slo_metric

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
