"""Service d'alertes pour les tentatives d'accès non autorisées via IP whitelist.

Fournit des fonctions pour enregistrer et alerter sur les tentatives d'accès
refusées aux endpoints protégés par IP whitelist.
"""

import logging
import os
from datetime import UTC, datetime, timedelta
from typing import Any, Dict

import sentry_sdk
from flask import request

from ext import redis_client
from security.audit_log import AuditLogger

logger = logging.getLogger(__name__)

# Configuration du rate limiting pour les alertes Sentry
ALERT_RATE_LIMIT_MINUTES = int(os.getenv("IP_WHITELIST_ALERT_RATE_LIMIT_MINUTES", "5"))
REDIS_ALERT_KEY_PREFIX = "ip_whitelist:alert:"


def should_alert_for_ip(client_ip: str) -> bool:
    """Vérifie si une alerte Sentry doit être envoyée pour cette IP.

    Implémente un rate limiting pour éviter le spam d'alertes:
    - Maximum 1 alerte Sentry par IP toutes les X minutes (configurable)

    Args:
        client_ip: Adresse IP du client

    Returns:
        True si une alerte doit être envoyée, False sinon
    """
    if not redis_client:
        # Si Redis n'est pas disponible, utiliser un cache en mémoire simple
        # Note: Ce cache sera perdu au redémarrage,
        # mais c'est acceptable pour le rate limiting
        if not hasattr(should_alert_for_ip, "_memory_cache"):
            should_alert_for_ip._memory_cache = {}  # type: ignore[attr-defined]

        memory_cache = should_alert_for_ip._memory_cache  # type: ignore[attr-defined]
        now = datetime.now(UTC)

        # Vérifier si une alerte a déjà été envoyée récemment
        if client_ip in memory_cache:
            last_alert_time = memory_cache[client_ip]
            if now - last_alert_time < timedelta(minutes=ALERT_RATE_LIMIT_MINUTES):
                return False

        # Enregistrer le timestamp de cette alerte
        memory_cache[client_ip] = now
        return True

    # Utiliser Redis pour le rate limiting (meilleure solution pour multi-workers)
    try:
        alert_key = f"{REDIS_ALERT_KEY_PREFIX}{client_ip}"
        exists = redis_client.exists(alert_key)

        if exists:
            # Une alerte a déjà été envoyée récemment
            return False

        # Enregistrer que nous envoyons une alerte maintenant
        redis_client.setex(alert_key, ALERT_RATE_LIMIT_MINUTES * 60, "1")
        return True

    except Exception as e:
        logger.warning(
            "[IP Whitelist Alerts] Erreur vérification rate limit Redis: %s", e
        )
        # En cas d'erreur Redis, autoriser l'alerte (fail-open)
        return True


def send_ip_whitelist_alert(
    client_ip: str,
    endpoint: str,
    method: str,
    user_agent: str | None = None,
    user_id: int | None = None,
) -> None:
    """Envoie une alerte pour une tentative d'accès non autorisée via IP whitelist.

    Cette fonction:
    1. Enregistre l'événement dans l'audit log
    2. Envoie une alerte Sentry (avec rate limiting)

    Args:
        client_ip: Adresse IP du client
        endpoint: Endpoint tenté (request.path)
        method: Méthode HTTP (GET, POST, etc.)
        user_agent: User-Agent de la requête (optionnel)
        user_id: ID utilisateur si authentifié (optionnel)
    """
    try:
        # Collecter les informations de la requête
        headers_to_log = {}
        if request:
            # Headers pertinents pour la sécurité
            security_headers = [
                "X-Forwarded-For",
                "X-Real-IP",
                "X-Forwarded-Proto",
                "Origin",
                "Referer",
            ]
            for header_name in security_headers:
                header_value = request.headers.get(header_name)
                if header_value:
                    headers_to_log[header_name] = header_value

        # 1. Enregistrer dans l'audit log (toujours, pas de rate limiting)
        audit_details = {
            "endpoint": endpoint,
            "method": method,
            "user_agent": user_agent,
            "headers": headers_to_log,
        }

        # Utiliser log_action directement pour inclure user_agent
        AuditLogger.log_action(
            action_type="ip_whitelist_denied",
            action_category="security",
            user_id=user_id,
            user_type="system",
            result_status="high",
            action_details=audit_details,
            ip_address=client_ip,
            user_agent=user_agent,
        )

        logger.info(
            (
                "[IP Whitelist Alerts] ✅ Événement enregistré dans audit log: "
                "IP=%s, endpoint=%s, method=%s"
            ),
            client_ip,
            endpoint,
            method,
        )

        # 2. Envoyer une alerte Sentry (avec rate limiting)
        if should_alert_for_ip(client_ip):
            # Construire le message d'alerte
            alert_message = (
                f"Tentative d'accès non autorisée via IP whitelist: "
                f"IP {client_ip} a tenté d'accéder à {method} {endpoint}"
            )

            # Tags pour faciliter le filtrage dans Sentry
            tags = {
                "security_event": "ip_whitelist_denied",
                "ip_address": client_ip,
                "endpoint": endpoint,
                "method": method,
            }

            # Context additionnel
            context: Dict[str, Any] = {
                "ip_address": client_ip,
                "endpoint": endpoint,
                "method": method,
                "user_agent": user_agent,
                "headers": headers_to_log,
            }
            if user_id:
                context["user_id"] = user_id

            # Envoyer l'alerte Sentry
            sentry_sdk.capture_message(
                alert_message,
                level="warning",  # "warning" car suspect mais pas forcément critique
                tags=tags,
                contexts={"request": context},
            )

            logger.warning(
                "[IP Whitelist Alerts] 🚨 Alerte Sentry envoyée: IP=%s, endpoint=%s",
                client_ip,
                endpoint,
            )
        else:
            logger.debug(
                "[IP Whitelist Alerts] ⏸️ Alerte Sentry ignorée (rate limit): IP=%s",
                client_ip,
            )

    except Exception as e:
        # Ne pas faire échouer la requête si l'alerte échoue
        logger.exception(
            "[IP Whitelist Alerts] ❌ Erreur lors de l'envoi d'alerte: %s", e
        )
