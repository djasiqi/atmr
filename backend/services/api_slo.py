"""Service Level Objectives (SLO) pour les routes API critiques.

Ce module définit les SLO pour chaque route critique de l'application,
permet de suivre les violations (breaches) et expose des métriques Prometheus.

Usage:
    from services.api_slo import get_slo_target, record_slo_metric

    # Dans une route
    slo = get_slo_target("/api/bookings")
    # ... exécuter la route ...
    record_slo_metric("/api/bookings", duration_seconds=0.350, status_code=200)
"""

from dataclasses import dataclass
from typing import Dict, Optional

# Import optionnel prometheus_client (peut ne pas être installé en dev)
try:
    from prometheus_client import Counter, Histogram

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = None
    Histogram = None


@dataclass
class APISLOTarget:
    """Définition d'un SLO pour une route API.

    Attributes:
        endpoint: Pattern de l'endpoint (ex: "/api/bookings" ou "/api/bookings/:id")
        latency_p95_max_ms: Latence p95 maximale en millisecondes
        error_rate_max: Taux d'erreurs maximal (ex: 0.01 = 1%)
        availability_min: Disponibilité minimale (ex: 0.99 = 99%)
        method: Méthode HTTP (optionnel, si None, s'applique à toutes les méthodes)
    """

    endpoint: str
    latency_p95_max_ms: int
    error_rate_max: float
    availability_min: float
    method: Optional[str] = None


# SLO par endpoint critique
API_SLOS: Dict[str, APISLOTarget] = {
    # Routes Bookings (critiques - utilisation fréquente)
    "/api/bookings": APISLOTarget(
        endpoint="/api/bookings",
        latency_p95_max_ms=500,
        error_rate_max=0.01,  # 1%
        availability_min=0.99,  # 99%
    ),
    "/api/bookings/:id": APISLOTarget(
        endpoint="/api/bookings/:id",
        latency_p95_max_ms=300,
        error_rate_max=0.01,
        availability_min=0.99,
    ),
    # Routes Companies (profil utilisateur)
    "/api/companies/me": APISLOTarget(
        endpoint="/api/companies/me",
        latency_p95_max_ms=300,
        error_rate_max=0.01,
        availability_min=0.99,
    ),
    # Routes Auth (critique pour l'accès)
    "/api/auth/login": APISLOTarget(
        endpoint="/api/auth/login",
        latency_p95_max_ms=500,
        error_rate_max=0.02,  # 2% toléré (tentatives invalides)
        availability_min=0.995,  # 99.5%
    ),
    "/api/auth/register": APISLOTarget(
        endpoint="/api/auth/register",
        latency_p95_max_ms=1000,  # Plus long (création compte)
        error_rate_max=0.02,
        availability_min=0.99,
    ),
    # Routes Dispatch (critique métier)
    "/api/dispatch/run": APISLOTarget(
        endpoint="/api/dispatch/run",
        latency_p95_max_ms=5000,  # 5s (calcul complexe)
        error_rate_max=0.05,  # 5% (traitement complexe)
        availability_min=0.95,  # 95% (moins critique que les routes clients)
    ),
    "/api/dispatch/status": APISLOTarget(
        endpoint="/api/dispatch/status",
        latency_p95_max_ms=200,
        error_rate_max=0.01,
        availability_min=0.99,
    ),
    # Routes Drivers (mise à jour fréquente)
    "/api/drivers": APISLOTarget(
        endpoint="/api/drivers",
        latency_p95_max_ms=400,
        error_rate_max=0.01,
        availability_min=0.99,
    ),
    # Routes Health (critique pour monitoring)
    "/api/health": APISLOTarget(
        endpoint="/api/health",
        latency_p95_max_ms=100,
        error_rate_max=0.001,  # 0.1% (doit être très fiable)
        availability_min=0.999,  # 99.9%
    ),
    "/api/ready": APISLOTarget(
        endpoint="/api/ready",
        latency_p95_max_ms=200,
        error_rate_max=0.001,
        availability_min=0.999,
    ),
}


def get_slo_target(endpoint: str, method: Optional[str] = None) -> Optional[APISLOTarget]:
    """Récupère la définition SLO pour un endpoint.

    Args:
        endpoint: Endpoint normalisé (ex: "/api/bookings/:id")
        method: Méthode HTTP (optionnel)

    Returns:
        APISLOTarget si trouvé, None sinon
    """
    # Chercher une correspondance exacte
    if endpoint in API_SLOS:
        slo = API_SLOS[endpoint]
        # Si une méthode est spécifiée dans le SLO, vérifier correspondance
        if slo.method and method and slo.method != method:
            return None
        return slo

    # Chercher une correspondance par préfixe (ex: "/api/bookings/:id" -> "/api/bookings")
    for slo_endpoint, slo in API_SLOS.items():
        if endpoint.startswith(slo_endpoint) and (not slo.method or slo.method == method):
            return slo

    return None


# Métriques Prometheus pour SLO breaches
if PROMETHEUS_AVAILABLE and Counter is not None and Histogram is not None:
    SLO_LATENCY_BREACH = Counter(
        "api_slo_latency_breach_total",
        "Nombre total de violations de latence SLO",
        ["endpoint", "method"],
    )

    SLO_ERROR_BREACH = Counter(
        "api_slo_error_breach_total",
        "Nombre total de violations de taux d'erreurs SLO",
        ["endpoint", "method"],
    )

    SLO_AVAILABILITY_BREACH = Counter(
        "api_slo_availability_breach_total",
        "Nombre total de violations de disponibilité SLO",
        ["endpoint", "method"],
    )

    # Histogram pour calculer p95/p99 (utilise les buckets du middleware metrics)
    SLO_LATENCY_HISTOGRAM = Histogram(
        "api_slo_request_duration_seconds",
        "Latence des requêtes pour calcul SLO (p95/p99)",
        ["endpoint", "method", "status"],
        buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )
else:
    SLO_LATENCY_BREACH = None
    SLO_ERROR_BREACH = None
    SLO_AVAILABILITY_BREACH = None
    SLO_LATENCY_HISTOGRAM = None


def record_slo_metric(
    endpoint: str,
    duration_seconds: float,
    status_code: int,
    method: str = "GET",
) -> None:
    """Enregistre une métrique pour le calcul de SLO.

    Cette fonction vérifie si la requête viole les SLO définis et incrémente
    les compteurs de breach Prometheus si nécessaire.

    Args:
        endpoint: Endpoint normalisé (ex: "/api/bookings/:id")
        duration_seconds: Durée de la requête en secondes
        status_code: Code HTTP de la réponse
        method: Méthode HTTP (GET, POST, etc.)
    """
    if not PROMETHEUS_AVAILABLE:
        return

    # Récupérer le SLO target
    slo = get_slo_target(endpoint, method)
    if not slo:
        # Pas de SLO défini pour cet endpoint
        return

    # Enregistrer dans l'histogram pour calcul p95/p99
    if SLO_LATENCY_HISTOGRAM:
        SLO_LATENCY_HISTOGRAM.labels(
            endpoint=endpoint,
            method=method,
            status=str(status_code),
        ).observe(duration_seconds)

    # Vérifier violation de latence
    latency_ms = duration_seconds * 1000
    if latency_ms > slo.latency_p95_max_ms and SLO_LATENCY_BREACH:
        SLO_LATENCY_BREACH.labels(
            endpoint=endpoint,
            method=method,
        ).inc()

    # Vérifier violation de taux d'erreurs (5xx = erreur serveur)
    HTTP_STATUS_ERROR_THRESHOLD = 500
    ERROR_RATE_MAX_THRESHOLD = 1.0
    is_error = status_code >= HTTP_STATUS_ERROR_THRESHOLD
    if is_error and slo.error_rate_max < ERROR_RATE_MAX_THRESHOLD and SLO_ERROR_BREACH:
        # Seulement si on suit les erreurs
        SLO_ERROR_BREACH.labels(
            endpoint=endpoint,
            method=method,
        ).inc()


def normalize_endpoint(endpoint: str) -> str:
    """Normalise un endpoint pour correspondre aux patterns SLO.

    Remplace les IDs numériques par :id pour regrouper les routes.
    Ex: /api/bookings/123 -> /api/bookings/:id

    Args:
        endpoint: Endpoint brut (ex: "/api/bookings/123")

    Returns:
        Endpoint normalisé (ex: "/api/bookings/:id")
    """
    import re

    # Pattern: /api/resource/123 ou /api/resource/123/subresource/456
    normalized = re.sub(r"/\d+(?=/|$)", "/:id", endpoint)

    # Limiter longueur (éviter labels trop longs)
    MAX_ENDPOINT_LENGTH = 100
    if len(normalized) > MAX_ENDPOINT_LENGTH:
        normalized = normalized[:MAX_ENDPOINT_LENGTH] + "..."

    return normalized
