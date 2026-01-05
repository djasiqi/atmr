"""✅ S3: Service d'alertes de sécurité pour détecter les tentatives d'accès non autorisé répétées.

Détecte les patterns suspects et génère des alertes pour :
- Tentatives de login échouées répétées depuis la même IP
- Tentatives d'accès non autorisé répétées (401/403)
- Patterns de brute force
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from ext import redis_client

logger = logging.getLogger(__name__)

# Configuration
ALERT_THRESHOLD_LOGIN_FAILURES = int(
    os.getenv("ALERT_THRESHOLD_LOGIN_FAILURES", "5")
)  # Nombre de tentatives échouées avant alerte
ALERT_THRESHOLD_UNAUTHORIZED = int(
    os.getenv("ALERT_THRESHOLD_UNAUTHORIZED", "10")
)  # Nombre de 401/403 avant alerte
ALERT_WINDOW_MINUTES = int(
    os.getenv("ALERT_WINDOW_MINUTES", "15")
)  # Fenêtre de temps pour détecter les patterns (15 minutes)
ALERT_COOLDOWN_MINUTES = int(
    os.getenv("ALERT_COOLDOWN_MINUTES", "60")
)  # Cooldown entre alertes pour la même IP (1 heure)


class SecurityAlertService:
    """Service pour détecter et alerter sur les tentatives d'accès non autorisé répétées."""

    @staticmethod
    def _get_redis_client() -> Any | None:
        """Récupère un client Redis pour le stockage des alertes."""
        if redis_client:
            try:
                redis_client.ping()
                return redis_client
            except Exception:
                logger.debug("[SecurityAlerts] Redis unavailable")
        return None

    @staticmethod
    def _get_alert_key(ip_address: str, alert_type: str) -> str:
        """Génère une clé Redis pour une alerte.

        Args:
            ip_address: Adresse IP
            alert_type: Type d'alerte ("login_failures", "unauthorized_access")

        Returns:
            Clé Redis
        """
        return f"security:alert:{alert_type}:{ip_address}"

    @staticmethod
    def _get_cooldown_key(ip_address: str, alert_type: str) -> str:
        """Génère une clé Redis pour le cooldown d'alerte.

        Args:
            ip_address: Adresse IP
            alert_type: Type d'alerte

        Returns:
            Clé Redis
        """
        return f"security:alert:cooldown:{alert_type}:{ip_address}"

    @staticmethod
    def record_login_failure(ip_address: str, email: str | None = None) -> bool:
        """✅ S3: Enregistre une tentative de login échouée et vérifie si une alerte doit être générée.

        Args:
            ip_address: Adresse IP de la tentative
            email: Email utilisé (optionnel, masqué dans les logs)

        Returns:
            True si une alerte a été générée, False sinon
        """
        redis_client_alert = SecurityAlertService._get_redis_client()
        if not redis_client_alert:
            # Si Redis n'est pas disponible, on ne peut pas tracker les alertes
            logger.debug("[SecurityAlerts] Redis unavailable, skipping alert tracking")
            return False

        try:
            alert_key = SecurityAlertService._get_alert_key(
                ip_address, "login_failures"
            )
            cooldown_key = SecurityAlertService._get_cooldown_key(
                ip_address, "login_failures"
            )

            # Vérifier si on est en cooldown
            if redis_client_alert.exists(cooldown_key):
                logger.debug(
                    "[SecurityAlerts] IP %s en cooldown pour login_failures", ip_address
                )
                return False

            # Incrémenter le compteur avec TTL
            count = redis_client_alert.incr(alert_key)
            redis_client_alert.expire(alert_key, ALERT_WINDOW_MINUTES * 60)

            # Vérifier si le seuil est atteint
            if count >= ALERT_THRESHOLD_LOGIN_FAILURES:
                # Générer l'alerte
                masked_email = email[:3] + "***" if email else "unknown"
                logger.warning(
                    "[SecurityAlerts] ⚠️ ALERTE: %d tentatives de login échouées depuis %s (email: %s)",
                    count,
                    ip_address,
                    masked_email,
                )

                # Enregistrer l'alerte dans AuditLog
                try:
                    from security.audit_log import AuditLogger

                    AuditLogger.log_action(
                        action_type="security_alert",
                        action_category="security",
                        user_type="system",
                        result_status="alert",
                        result_message=f"Tentatives de login échouées répétées depuis {ip_address}",
                        action_details={
                            "alert_type": "login_failures",
                            "ip_address": ip_address,
                            "failure_count": count,
                            "email_attempted": masked_email,
                            "threshold": ALERT_THRESHOLD_LOGIN_FAILURES,
                        },
                        ip_address=ip_address,
                    )
                except Exception as e:
                    logger.error("[SecurityAlerts] Failed to log alert: %s", e)

                # Mettre en cooldown pour éviter spam d'alertes
                redis_client_alert.setex(cooldown_key, ALERT_COOLDOWN_MINUTES * 60, "1")

                # Incrémenter métrique Prometheus
                try:
                    from security.security_metrics import (
                        security_unauthorized_access_total,
                    )

                    security_unauthorized_access_total.labels(
                        status_code="401", endpoint="/auth/login"
                    ).inc()
                except Exception:
                    pass  # Ne pas bloquer si métriques indisponibles

                return True

            return False

        except Exception as e:
            logger.error("[SecurityAlerts] Error recording login failure: %s", e)
            return False

    @staticmethod
    def record_unauthorized_access(
        ip_address: str, endpoint: str, status_code: int, user_id: int | None = None
    ) -> bool:
        """✅ S3: Enregistre une tentative d'accès non autorisé et vérifie si une alerte doit être générée.

        Args:
            ip_address: Adresse IP de la tentative
            endpoint: Endpoint accédé
            status_code: Code de statut HTTP (401 ou 403)
            user_id: ID utilisateur si authentifié (optionnel)

        Returns:
            True si une alerte a été générée, False sinon
        """
        if status_code not in (401, 403):
            return False

        redis_client_alert = SecurityAlertService._get_redis_client()
        if not redis_client_alert:
            logger.debug("[SecurityAlerts] Redis unavailable, skipping alert tracking")
            return False

        try:
            alert_key = SecurityAlertService._get_alert_key(
                ip_address, "unauthorized_access"
            )
            cooldown_key = SecurityAlertService._get_cooldown_key(
                ip_address, "unauthorized_access"
            )

            # Vérifier si on est en cooldown
            if redis_client_alert.exists(cooldown_key):
                logger.debug(
                    "[SecurityAlerts] IP %s en cooldown pour unauthorized_access",
                    ip_address,
                )
                return False

            # Incrémenter le compteur avec TTL
            count = redis_client_alert.incr(alert_key)
            redis_client_alert.expire(alert_key, ALERT_WINDOW_MINUTES * 60)

            # Vérifier si le seuil est atteint
            if count >= ALERT_THRESHOLD_UNAUTHORIZED:
                # Générer l'alerte
                logger.warning(
                    "[SecurityAlerts] ⚠️ ALERTE: %d tentatives d'accès non autorisé depuis %s (endpoint: %s, status: %d)",
                    count,
                    ip_address,
                    endpoint,
                    status_code,
                )

                # Enregistrer l'alerte dans AuditLog
                try:
                    from security.audit_log import AuditLogger

                    AuditLogger.log_action(
                        action_type="security_alert",
                        action_category="security",
                        user_id=user_id,
                        user_type="unknown" if user_id is None else "authenticated",
                        result_status="alert",
                        result_message=f"Tentatives d'accès non autorisé répétées depuis {ip_address}",
                        action_details={
                            "alert_type": "unauthorized_access",
                            "ip_address": ip_address,
                            "endpoint": endpoint,
                            "status_code": status_code,
                            "failure_count": count,
                            "threshold": ALERT_THRESHOLD_UNAUTHORIZED,
                        },
                        ip_address=ip_address,
                    )
                except Exception as e:
                    logger.error("[SecurityAlerts] Failed to log alert: %s", e)

                # Mettre en cooldown
                redis_client_alert.setex(cooldown_key, ALERT_COOLDOWN_MINUTES * 60, "1")

                # Incrémenter métrique Prometheus
                try:
                    from security.security_metrics import (
                        security_unauthorized_access_total,
                    )

                    security_unauthorized_access_total.labels(
                        status_code=str(status_code), endpoint=endpoint
                    ).inc()
                except Exception:
                    pass

                return True

            return False

        except Exception as e:
            logger.error("[SecurityAlerts] Error recording unauthorized access: %s", e)
            return False

    @staticmethod
    def get_suspicious_ips() -> List[Dict[str, Any]]:
        """✅ S3: Récupère la liste des IPs suspectes actuellement trackées.

        Returns:
            Liste de dicts avec informations sur les IPs suspectes
        """
        redis_client_alert = SecurityAlertService._get_redis_client()
        if not redis_client_alert:
            return []

        try:
            suspicious_ips: List[Dict[str, Any]] = []

            # Chercher toutes les clés d'alertes
            for alert_type in ["login_failures", "unauthorized_access"]:
                pattern = f"security:alert:{alert_type}:*"
                keys = redis_client_alert.keys(pattern)

                for key in keys:
                    # Extraire l'IP de la clé
                    ip_address = key.decode("utf-8").split(":")[-1]
                    count = redis_client_alert.get(key)
                    ttl = redis_client_alert.ttl(key)

                    if count:
                        suspicious_ips.append(
                            {
                                "ip_address": ip_address,
                                "alert_type": alert_type,
                                "count": int(count),
                                "ttl_seconds": ttl,
                            }
                        )

            return suspicious_ips

        except Exception as e:
            logger.error("[SecurityAlerts] Error getting suspicious IPs: %s", e)
            return []
