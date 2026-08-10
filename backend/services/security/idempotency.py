"""Service pour gérer l'idempotence des requêtes API.

Permet d'éviter les doublons de requêtes en utilisant une clé d'idempotence.
"""

import json
import logging
from typing import Any

from flask import request

from ext import redis_client

logger = logging.getLogger(__name__)


class IdempotencyService:
    """Service pour gérer l'idempotence des requêtes."""

    @staticmethod
    def check_key(key: str, ttl: int = 86400) -> tuple[bool, dict[str, Any] | None]:
        """Vérifie si une clé d'idempotence existe et retourne la réponse précédente.

        Args:
            key: Clé d'idempotence (généralement un UUID)
            ttl: Time-to-live en secondes (défaut: 24h) - non utilisé actuellement

        Returns:
            Tuple (exists, previous_response):
            - exists: True si la clé existe
            - previous_response: Réponse précédente si existe, None sinon
        """
        _ = ttl
        if not redis_client:
            logger.warning("Redis non disponible, idempotency désactivé")
            return False, None

        try:
            redis_key = f"idempotency:{key}"
            cached_response = redis_client.get(redis_key)

            if cached_response:
                try:
                    response_data = json.loads(cached_response)
                    logger.info("Idempotency key trouvée: %s", key)
                    return True, response_data
                except json.JSONDecodeError:
                    logger.warning("Erreur parsing réponse idempotency: %s", key)
                    return False, None

            return False, None
        except Exception as e:
            logger.error("Erreur lors de la vérification idempotency: %s", e)
            # En cas d'erreur, permettre la requête (fail-open)
            return False, None

    @staticmethod
    def store_response(
        key: str, response: dict[str, Any], status_code: int, ttl: int = 86400
    ) -> None:
        """Stocke une réponse pour une clé d'idempotence donnée.

        Args:
            key: Clé d'idempotence
            response: Réponse à stocker
            status_code: Code de statut HTTP
            ttl: Time-to-live en secondes (défaut: 24h)
        """
        if not redis_client:
            logger.warning("Redis non disponible, impossible de stocker idempotency")
            return

        try:
            redis_key = f"idempotency:{key}"
            response_data = {
                "response": response,
                "status_code": status_code,
            }
            redis_client.setex(redis_key, ttl, json.dumps(response_data, default=str))
            logger.debug("Réponse idempotency stockée: %s (TTL: %ds)", key, ttl)
        except Exception as e:
            logger.error("Erreur lors du stockage idempotency: %s", e)
            # Ne pas bloquer la réponse en cas d'erreur

    @staticmethod
    def get_idempotency_key_from_request() -> str | None:
        """Extrait la clé d'idempotence depuis les headers de la requête.

        Returns:
            Clé d'idempotence ou None si absente
        """
        return request.headers.get("Idempotency-Key") or request.headers.get(
            "X-Idempotency-Key"
        )
