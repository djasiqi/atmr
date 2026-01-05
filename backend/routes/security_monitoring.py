"""✅ S3: Routes pour le monitoring de sécurité.

Endpoints pour consulter les alertes de sécurité et les métriques.
"""

import logging

from flask_jwt_extended import jwt_required  # pyright: ignore[reportMissingImports]
from flask_restx import Namespace, Resource  # pyright: ignore[reportMissingImports]

from ext import limiter, role_required
from security.security_alerts import SecurityAlertService
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)

security_monitoring_ns = Namespace(
    "security-monitoring", description="Monitoring de sécurité"
)


@security_monitoring_ns.route("/alerts/suspicious-ips")
class SuspiciousIPsResource(Resource):
    """✅ S3: Liste des IPs suspectes détectées."""

    @jwt_required()
    @role_required("ADMIN")
    @limiter.limit("50 per hour")  # ✅ S2: Rate limiting pour endpoint admin
    def get(self):
        """Récupère la liste des IPs suspectes actuellement trackées.

        Returns:
            Liste des IPs suspectes avec compteurs et TTL
        """
        try:
            suspicious_ips = SecurityAlertService.get_suspicious_ips()

            return {
                "suspicious_ips": suspicious_ips,
                "count": len(suspicious_ips),
            }, 200

        except Exception as e:
            logger.error("[SecurityMonitoring] Error getting suspicious IPs: %s", e)
            return APIErrorHandler.handle_exception(e, logger)
