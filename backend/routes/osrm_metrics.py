"""
Routes de monitoring pour l'état du client OSRM.
Exposé uniquement pour les admins système (ou en interne).
"""
import logging

from flask import jsonify
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource

from services.osrm_client import _osrm_circuit_breaker

logger = logging.getLogger(__name__)

ns_osrm_metrics = Namespace(
    "osrm-metrics",
    description="Métriques et état du client OSRM (timeouts, circuit-breaker)"
)


@ns_osrm_metrics.route("/status")
class OsrmStatus(Resource):
    @jwt_required()
    def get(self):
        """
        Retourne l'état actuel du circuit-breaker OSRM.

        Utile pour:
        - Diagnostiquer les timeouts OSRM
        - Vérifier si le circuit-breaker est ouvert
        - Monitorer le nombre d'échecs consécutifs
        """
        cb = _osrm_circuit_breaker
        state = "closed"
        if cb.is_open():  # type: ignore[attr-defined]
            state = "open"
        elif getattr(cb, '_failure_count', 0) > 0:
            state = "half_open"

        return jsonify({
            "status": "ok",
            "circuit_breaker": {
                "state": state,
                "failure_count": getattr(cb, '_failure_count', 0),
                "last_failure_time": getattr(cb, '_last_failure_time', 0),
                "threshold": cb.failure_threshold,  # type: ignore[attr-defined]
                "timeout_seconds": getattr(cb, 'timeout', 60),
            },
            "message": (
                "Circuit is OPEN - OSRM requests blocked temporarily"
                if state == "open"
                else "Circuit is CLOSED - OSRM operational"
            ),
        })


@ns_osrm_metrics.route("/reset")
class OsrmReset(Resource):
    @jwt_required()
    def post(self):
        """
        Réinitialise manuellement le circuit-breaker (force CLOSED).

        ⚠️ À utiliser uniquement pour forcer la réouverture après
        vérification manuelle que l'OSRM backend est opérationnel.
        """
        cb = _osrm_circuit_breaker
        cb._failure_count = 0  # type: ignore[attr-defined]
        cb._last_failure_time = 0  # type: ignore[attr-defined]
        logger.warning("[OSRM] Circuit-breaker réinitialisé manuellement via API")
        return jsonify({
            "status": "ok",
            "message": "Circuit-breaker reset to CLOSED state"
        })
