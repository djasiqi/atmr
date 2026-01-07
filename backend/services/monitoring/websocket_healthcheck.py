# backend/services/websocket_healthcheck.py

"""Service de health check pour Socket.IO."""

import logging
from datetime import UTC, datetime
from typing import Any, Dict

from ext import redis_client
from services.monitoring.websocket_metrics import ws_metrics

logger = logging.getLogger(__name__)

# Seuils pour déterminer le statut
LATENCY_OK_MS = 200
LATENCY_DEGRADED_MS = 500
RECONNECTION_RATE_THRESHOLD = 0.1  # 10% de reconnexions par heure


def check_websocket_health() -> Dict[str, Any]:
    """Vérifie la santé du service Socket.IO.

    Returns:
        Dict avec status ("ok"|"degraded"|"error"), latency_ms, handlers_ok, etc.
    """
    result: Dict[str, Any] = {
        "status": "ok",
        "latency_ms": 0.0,
        "handlers_ok": True,
        "redis_queue": "ok",
        "connections_active": 0,
        "last_check": datetime.now(UTC).isoformat(),
    }

    # 1. Vérifier les métriques existantes
    stats = ws_metrics.get_stats()
    connections_active = stats.get("connections", {}).get("active_total", 0)
    result["connections_active"] = connections_active

    # 2. Vérifier la latence heartbeat
    heartbeat_stats = stats.get("heartbeat", {}).get("latency_ms", {})
    avg_latency = heartbeat_stats.get("avg", 0.0)
    result["latency_ms"] = avg_latency

    # 3. Vérifier le taux de reconnexion
    reconnections_total = stats.get("reconnections", {}).get("total", 0)
    connections_total = stats.get("connections", {}).get("total", 0)

    # Calculer le taux de reconnexion (reconnexions / connexions totales)
    # Si on a plus de 10% de reconnexions, c'est un signe de problème
    reconnection_rate = (
        reconnections_total / connections_total if connections_total > 0 else 0.0
    )

    # 4. Test ping/pong (si socketio est disponible)
    handlers_ok = True
    ping_latency = 0.0

    try:
        # Mesurer la latence d'un ping/pong simulé
        # Note: On ne peut pas vraiment tester un ping/pong réel sans connexion client
        # On utilise donc les métriques existantes comme proxy
        ping_latency = avg_latency
        result["ping_latency_ms"] = ping_latency

    except Exception as e:
        logger.warning("Erreur lors du test Socket.IO: %s", e, exc_info=True)
        handlers_ok = False
        result["handlers_ok"] = False
        result["error"] = str(e)

    # 5. Vérifier Redis (si disponible)
    redis_status = "ok"
    try:
        if redis_client:
            redis_client.ping()
            redis_status = "ok"
        else:
            redis_status = "not_configured"
    except Exception as e:
        logger.warning("Erreur Redis lors du health check: %s", e)
        redis_status = f"error: {e!s}"

    result["redis_queue"] = redis_status

    # 6. Déterminer le statut global
    status = "ok"

    # Erreur si :
    # - Aucune connexion active ET aucune connexion totale (service jamais utilisé)
    # - Handlers KO
    # - Latence très élevée (> 500ms)
    if not handlers_ok:
        status = "error"
    elif connections_total == 0 and connections_active == 0:
        # Service jamais utilisé - on considère ça comme OK (pas d'erreur)
        status = "ok"
    elif ping_latency > LATENCY_DEGRADED_MS:
        status = "error"
    elif (
        ping_latency > LATENCY_OK_MS or reconnection_rate > RECONNECTION_RATE_THRESHOLD
    ):
        status = "degraded"

    result["status"] = status
    result["reconnection_rate"] = reconnection_rate
    result["reconnections_total"] = reconnections_total
    result["connections_total"] = connections_total

    return result

