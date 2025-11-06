# backend/routes/osrm_health.py
"""✅ Endpoint de santé OSRM avec métriques cache et circuit breaker."""

import logging
from typing import Any, Dict, Tuple

from flask_restx import Namespace, Resource

from services.osrm_client import _osrm_circuit_breaker
from services.unified_dispatch.osrm_cache_metrics import (
    get_cache_metrics_dict,
)

logger = logging.getLogger(__name__)

# Constantes pour éviter les valeurs magiques
CACHE_HIT_RATE_DEGRADED_THRESHOLD = 0.5  # Seuil de hit rate pour considérer le cache comme dégradé

osrm_health_ns = Namespace(
    "osrm",
    description="OSRM health and metrics",
    path="/osrm"
)


@osrm_health_ns.route("/health")
class OsrmHealth(Resource):
    """Endpoint de santé OSRM avec métriques cache et circuit breaker."""
    
    def get(self) -> Tuple[Dict[str, Any], int]:
        """Retourne l'état de santé OSRM.
        
        Returns:
            Tuple contenant:
            - dict: {
                "status": "healthy" | "degraded" | "unhealthy",
                "circuit_breaker": {
                    "state": "CLOSED" | "OPEN" | "HALF_OPEN",
                    "failure_count": int,
                    "last_failure_time": float | None
                },
                "cache": {
                    "hit_rate": float,
                    "hits": int,
                    "misses": int,
                    "bypass_count": int
                },
                "timeout_adaptive": {
                    "enabled": bool,
                    "base_timeout": int,
                    "max_timeout": int
                }
            }
            - int: Status code HTTP (200 pour succès, 500 pour erreur)
        """
        try:
            # État circuit breaker
            cb_state = _osrm_circuit_breaker.state
            cb_failures = _osrm_circuit_breaker.failure_count
            cb_last_failure = _osrm_circuit_breaker.last_failure_time
            
            # Métriques cache
            cache_metrics = get_cache_metrics_dict()
            cache_hit_rate = cache_metrics.get("hit_rate", 0.0)
            
            # Déterminer le statut global
            status = "healthy"
            if cb_state == "OPEN":
                status = "unhealthy"
            elif cb_state == "HALF_OPEN" or cache_hit_rate < CACHE_HIT_RATE_DEGRADED_THRESHOLD:
                status = "degraded"
            
            # Timeout adaptatif config
            import os
            base_timeout = int(os.getenv("UD_OSRM_TIMEOUT", "45"))
            max_timeout = 120  # Max timeout adaptatif
            
            return {
                "status": status,
                "circuit_breaker": {
                    "state": cb_state,
                    "failure_count": cb_failures,
                    "last_failure_time": cb_last_failure,
                    "failure_threshold": _osrm_circuit_breaker.failure_threshold,
                    "timeout_duration": _osrm_circuit_breaker.timeout_duration,
                },
                "cache": {
                    "hit_rate": cache_hit_rate,
                    "hits": cache_metrics.get("hits", 0),
                    "misses": cache_metrics.get("misses", 0),
                    "bypass_count": cache_metrics.get("bypass_count", 0),
                    "total": cache_metrics.get("total", 0),
                },
                "timeout_adaptive": {
                    "enabled": True,
                    "base_timeout": base_timeout,
                    "max_timeout": max_timeout,
                    "formula": "15s base + 0.5s/point (max 120s)",
                },
            }, 200
            
        except Exception as e:
            logger.exception("[OsrmHealth] Erreur récupération health: %s", e)
            return {
                "status": "error",
                "error": str(e),
            }, 500

