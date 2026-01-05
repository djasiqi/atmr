"""
Service pour gérer la révocation des access tokens.

Note: Les access tokens expirent rapidement (1-2h), donc la révocation
est moins critique que pour les refresh tokens. Cependant, elle permet
de révoquer immédiatement un token lors d'un logout ou d'une action
administrative.
"""

import logging

import redis  # pyright: ignore[reportMissingImports]
from flask import current_app  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class AccessTokenService:
    """Service pour gérer la révocation des access tokens."""

    def __init__(self) -> None:
        """Initialise le service avec une connexion Redis."""
        redis_url = current_app.config.get("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.blacklist_prefix = "blacklisted_access_token:"

    def revoke_token(self, token_jti: str, expires_in: int) -> None:
        """Ajoute un token à la blacklist.

        Args:
            token_jti: Le JWT ID (jti claim) du token à révoquer
            expires_in: Temps restant avant expiration en secondes (TTL)
        """
        if expires_in <= 0:
            logger.warning(
                "Tentative de révocation d'un token déjà expiré: jti=%s", token_jti
            )
            return

        self.redis_client.setex(
            f"{self.blacklist_prefix}{token_jti}",
            expires_in,
            "revoked",
        )
        logger.info("Access token révoqué: jti=%s, ttl=%s", token_jti, expires_in)

        # ✅ PHASE 3: Métrique Prometheus
        try:
            from security.security_metrics import tokens_revoked_total

            tokens_revoked_total.labels(token_type="access_token").inc()
        except Exception:
            pass  # Ne pas bloquer si métriques indisponibles

    def is_token_revoked(self, token_jti: str) -> bool:
        """Vérifie si un token est révoqué.

        Args:
            token_jti: Le JWT ID (jti claim) du token à vérifier

        Returns:
            True si le token est révoqué, False sinon
        """
        return bool(self.redis_client.exists(f"{self.blacklist_prefix}{token_jti}"))
