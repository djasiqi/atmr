"""
Services pour gérer l'authentification JWT (access tokens et refresh tokens).

Ce module consolide les services de gestion des tokens JWT :
- AccessTokenService : Révocation des access tokens via blacklist Redis
- RefreshTokenService : Gestion des refresh tokens avec rotation et limitation

## Migration B2 (7 janvier 2025)

Ce fichier consolide :
- `services/access_token_service.py` → AccessTokenService
- `services/refresh_token_service.py` → RefreshTokenService

## Usage

```python
# Nouveau (recommandé)
from services.security.authentication import AccessTokenService, RefreshTokenService

# Ancien (DEPRECATED)
# from services.access_token_service import AccessTokenService
# from services.refresh_token_service import RefreshTokenService
```

## Architecture

- Les access tokens expirent rapidement (1-2h) et sont révoqués via blacklist Redis
- Les refresh tokens ont une durée de vie longue (30 jours) et nécessitent une gestion
  plus stricte avec rotation automatique et limitation du nombre de tokens actifs

---

**Version :** 1.0.0
**Date :** 7 janvier 2025
"""

import hashlib
import logging

import redis  # pyright: ignore[reportMissingImports]
from flask import current_app  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


class AccessTokenService:
    """Service pour gérer la révocation des access tokens.

    Note: Les access tokens expirent rapidement (1-2h), donc la révocation
    est moins critique que pour les refresh tokens. Cependant, elle permet
    de révoquer immédiatement un token lors d'un logout ou d'une action
    administrative.
    """

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

        # ✅ Métrique Prometheus
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


class RefreshTokenService:
    """Service pour gérer les refresh tokens avec rotation et limitation.

    Ce service utilise Redis pour stocker les tokens actifs et révoqués,
    permettant une rotation automatique des refresh tokens et une limitation
    du nombre de tokens actifs par utilisateur.
    """

    def __init__(self) -> None:
        """Initialise le service avec une connexion Redis."""
        redis_url = current_app.config.get("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.revoked_tokens_prefix = "revoked_refresh_token:"
        self.active_tokens_prefix = "active_refresh_token:"

    def is_token_valid(self, token: str, user_id: int | None = None) -> bool:
        """Vérifie si un token est valide (non révoqué et appartenant à l'utilisateur).

        Args:
            token: Le refresh token JWT en clair
            user_id: ID de l'utilisateur (optionnel, pour vérification supplémentaire)

        Returns:
            True si le token est valide, False sinon

        Note:
            Si Redis est indisponible, cette fonction retourne True par défaut
            pour permettre la validation JWT standard (fallback gracieux).
            Cela évite de déconnecter tous les utilisateurs lors de problèmes Redis.
        """
        token_hash = self._hash_token(token)

        try:
            # 1. Vérifier si le token est révoqué
            if self.redis_client.exists(f"{self.revoked_tokens_prefix}{token_hash}"):
                logger.warning("Token révoqué utilisé: user_id=%s", user_id)
                return False

            # 2. Vérifier si le token est dans les tokens actifs
            stored_user_id = self.redis_client.get(
                f"{self.active_tokens_prefix}{token_hash}"
            )

            # ✅ FALLBACK: Si Redis est indisponible ou données perdues,
            # permettre la validation JWT standard (ne pas bloquer l'utilisateur)
            if not stored_user_id:
                # Si Redis est disponible mais le token n'est pas dans les actifs,
                # vérifier si c'est une erreur Redis ou si le token n'existe vraiment pas
                try:
                    # Tester la connexion Redis avec un ping
                    self.redis_client.ping()
                    # Redis est disponible mais token pas dans actifs → invalide
                    logger.debug(
                        "Token non trouvé dans Redis actifs (Redis disponible): user_id=%s",
                        user_id,
                    )
                    return False
                except Exception:
                    # Redis indisponible → fallback gracieux
                    logger.warning(
                        "Redis indisponible lors validation token. Fallback gracieux activé (validation JWT uniquement): user_id=%s",
                        user_id,
                    )
                    return True  # Permettre validation JWT standard

            # 3. Si user_id fourni, vérifier si le token appartient à l'utilisateur
            if user_id is not None and int(stored_user_id) != user_id:
                logger.warning(
                    "Token utilisé par mauvais utilisateur: token_user_id=%s, "
                    "requested_user_id=%s",
                    stored_user_id,
                    user_id,
                )
                return False

            return True
        except Exception as redis_error:
            # ✅ FALLBACK GRACIEUX: Si Redis échoue, permettre validation JWT standard
            # Cela évite de déconnecter tous les utilisateurs lors de problèmes Redis
            logger.warning(
                "Erreur Redis lors validation token. Fallback gracieux activé (validation JWT uniquement): %s, user_id=%s",
                str(redis_error),
                user_id,
            )
            return True  # Permettre validation JWT standard

    def revoke_token(self, token: str) -> None:
        """Révoque un token (le marque comme invalide).

        Args:
            token: Le refresh token JWT en clair
        """
        token_hash = self._hash_token(token)

        # Obtenir l'expiration du token depuis la config
        try:
            refresh_expires_delta = current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
            ttl = int(refresh_expires_delta.total_seconds())
        except Exception:
            # Fallback: 30 jours en secondes
            ttl = 30 * 24 * 3600

        # Marquer comme révoqué avec TTL = durée de validité du token
        self.redis_client.setex(
            f"{self.revoked_tokens_prefix}{token_hash}",
            ttl,
            "revoked",
        )

        # Supprimer de la liste des tokens actifs
        stored_user_id = self.redis_client.get(
            f"{self.active_tokens_prefix}{token_hash}"
        )
        if stored_user_id:
            user_tokens_key = f"user_refresh_tokens:{stored_user_id}"
            self.redis_client.srem(user_tokens_key, token_hash)
            self.redis_client.delete(f"{self.active_tokens_prefix}{token_hash}")

        logger.info("Token révoqué: %s...", token_hash[:8])

        # ✅ Métrique Prometheus
        try:
            from security.security_metrics import tokens_revoked_total

            tokens_revoked_total.labels(token_type="refresh_token").inc()
        except Exception:
            pass  # Ne pas bloquer si métriques indisponibles

    def store_token(self, user_id: int, token: str) -> None:
        """Stocke un nouveau token actif.

        Args:
            user_id: ID de l'utilisateur propriétaire du token
            token: Le refresh token JWT en clair
        """
        token_hash = self._hash_token(token)

        # Obtenir l'expiration du token depuis la config
        try:
            refresh_expires_delta = current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
            ttl = int(refresh_expires_delta.total_seconds())
        except Exception:
            # Fallback: 30 jours en secondes
            ttl = 30 * 24 * 3600

        # Stocker le mapping token -> user_id
        self.redis_client.setex(
            f"{self.active_tokens_prefix}{token_hash}",
            ttl,
            str(user_id),
        )

        # Ajouter à la liste des tokens actifs de l'utilisateur
        user_tokens_key = f"user_refresh_tokens:{user_id}"
        self.redis_client.sadd(user_tokens_key, token_hash)
        self.redis_client.expire(user_tokens_key, ttl)

        logger.info("Token stocké: user_id=%s, hash=%s...", user_id, token_hash[:8])

    def revoke_all_user_tokens(self, user_id: int) -> None:
        """Révoque tous les tokens actifs d'un utilisateur (pour logout).

        Args:
            user_id: ID de l'utilisateur
        """
        user_tokens_key = f"user_refresh_tokens:{user_id}"
        token_hashes = self.redis_client.smembers(user_tokens_key)

        # Obtenir l'expiration du token depuis la config
        try:
            refresh_expires_delta = current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
            ttl = int(refresh_expires_delta.total_seconds())
        except Exception:
            # Fallback: 30 jours en secondes
            ttl = 30 * 24 * 3600

        for token_hash in token_hashes:
            # Marquer comme révoqué
            self.redis_client.setex(
                f"{self.revoked_tokens_prefix}{token_hash}",
                ttl,
                "revoked",
            )
            # Supprimer de la liste des actifs
            self.redis_client.delete(f"{self.active_tokens_prefix}{token_hash}")

        # Supprimer la liste des tokens actifs
        self.redis_client.delete(user_tokens_key)

        logger.info("Tous les tokens révoqués pour user_id=%s", user_id)

        # ✅ Métrique Prometheus
        try:
            from security.security_metrics import tokens_revoked_total

            # Incrémenter pour chaque token révoqué
            tokens_revoked_total.labels(token_type="refresh_token").inc(
                len(token_hashes)
            )
        except Exception:
            pass  # Ne pas bloquer si métriques indisponibles

    def get_active_token_count(self, user_id: int) -> int:
        """Retourne le nombre de tokens actifs pour un utilisateur.

        Args:
            user_id: ID de l'utilisateur

        Returns:
            Nombre de tokens actifs
        """
        user_tokens_key = f"user_refresh_tokens:{user_id}"
        return self.redis_client.scard(user_tokens_key)

    def limit_active_tokens(self, user_id: int, max_tokens: int = 5) -> None:
        """Limite le nombre de tokens actifs par utilisateur.

        Si le nombre de tokens actifs dépasse max_tokens, les tokens en excès
        sont révoqués (FIFO approximatif via SRANDMEMBER).

        Args:
            user_id: ID de l'utilisateur
            max_tokens: Nombre maximum de tokens actifs (défaut: 5)
        """
        user_tokens_key = f"user_refresh_tokens:{user_id}"
        count = self.redis_client.scard(user_tokens_key)

        if count > max_tokens:
            # Supprimer les tokens les plus anciens (FIFO approximatif)
            excess = count - max_tokens
            tokens_to_revoke = self.redis_client.srandmember(user_tokens_key, excess)

            # Si srandmember retourne une seule valeur au lieu d'une liste
            if not isinstance(tokens_to_revoke, list):
                tokens_to_revoke = [tokens_to_revoke] if tokens_to_revoke else []

            # Obtenir l'expiration du token depuis la config
            try:
                refresh_expires_delta = current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
                ttl = int(refresh_expires_delta.total_seconds())
            except Exception:
                # Fallback: 30 jours en secondes
                ttl = 30 * 24 * 3600

            for token_hash in tokens_to_revoke:
                # Marquer comme révoqué
                self.redis_client.setex(
                    f"{self.revoked_tokens_prefix}{token_hash}",
                    ttl,
                    "revoked",
                )
                # Supprimer de la liste des actifs
                self.redis_client.delete(f"{self.active_tokens_prefix}{token_hash}")
                self.redis_client.srem(user_tokens_key, token_hash)

            logger.info(
                "Tokens limités pour user_id=%s: %s -> %s", user_id, count, max_tokens
            )

    @staticmethod
    def _hash_token(token: str) -> str:
        """Hash un token pour stockage sécurisé.

        Args:
            token: Le refresh token JWT en clair

        Returns:
            Hash SHA256 du token (64 caractères hexadécimaux)
        """
        return hashlib.sha256(token.encode()).hexdigest()
