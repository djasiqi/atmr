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

from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, TypeVar

import redis
from flask import current_app

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _fix_wrongtype_and_retry(
    redis_client: redis.Redis,
    key: str,
    op: Callable[[], T],
    *,
    expected_type: str = "zset",
) -> T:
    """Exécute op(); en cas de WRONGTYPE Redis, supprime la clé et réessaie.

    Corrige la dette technique où user_refresh_tokens:* peut avoir un type
    incorrect (ex. STRING au lieu de ZSET) suite à migrations ou bugs passés.
    """
    try:
        return op()
    except redis.ResponseError as e:
        if "WRONGTYPE" not in str(e).upper():
            raise
        logger.warning(
            "Redis WRONGTYPE sur clé %s (attendu %s): suppression et retry. err=%s",
            key,
            expected_type,
            e,
        )
        redis_client.delete(key)
        return op()


class AccessTokenService:
    """Service pour gérer la révocation des access tokens.

    Note: Les access tokens expirent rapidement (1-2h), donc la révocation
    est moins critique que pour les refresh tokens. Cependant, elle permet
    de révoquer immédiatement un token lors d'un logout ou d'une action
    administrative.
    """

    def __init__(self) -> None:
        """Initialise le service avec une connexion Redis."""
        super().__init__()
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
        super().__init__()
        redis_url = current_app.config.get("REDIS_URL", "redis://127.0.0.1:6379/0")
        self.redis_client: Any = redis.from_url(redis_url, decode_responses=True)
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
                except Exception as redis_ping_err:
                    from security.refresh_token_service import (
                        RefreshStoreUnavailableError,
                        refresh_fail_closed_enabled,
                    )

                    if refresh_fail_closed_enabled():
                        logger.error(
                            "Redis indisponible lors validation refresh (fail-closed): user_id=%s",
                            user_id,
                        )
                        raise RefreshStoreUnavailableError(
                            "redis_unavailable"
                        ) from redis_ping_err
                    logger.warning(
                        "Redis indisponible lors validation token. Fallback gracieux activé (validation JWT uniquement): user_id=%s",
                        user_id,
                    )
                    return True

            # 3. Si user_id fourni, vérifier si le token appartient à l'utilisateur
            if user_id is not None and int(stored_user_id) != user_id:
                logger.warning(
                    (
                        "Token utilisé par mauvais utilisateur: token_user_id=%s, "
                        "requested_user_id=%s"
                    ),
                    stored_user_id,
                    user_id,
                )
                return False

            return True
        except Exception as redis_error:
            from security.refresh_token_service import (
                RefreshStoreUnavailableError,
                refresh_fail_closed_enabled,
            )

            if isinstance(redis_error, RefreshStoreUnavailableError):
                raise
            if refresh_fail_closed_enabled():
                logger.error(
                    "Erreur Redis refresh (fail-closed): %s, user_id=%s",
                    str(redis_error),
                    user_id,
                )
                raise RefreshStoreUnavailableError("redis_unavailable") from redis_error
            logger.warning(
                "Erreur Redis lors validation token. Fallback gracieux activé (validation JWT uniquement): %s, user_id=%s",
                str(redis_error),
                user_id,
            )
            return True

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

        stored_user_id = self.redis_client.get(
            f"{self.active_tokens_prefix}{token_hash}"
        )
        if stored_user_id:
            user_tokens_key = f"user_refresh_tokens:{stored_user_id}"

            def _zrem() -> Any:
                self.redis_client.zrem(user_tokens_key, token_hash)

            _fix_wrongtype_and_retry(self.redis_client, user_tokens_key, _zrem)
            self.redis_client.delete(f"{self.active_tokens_prefix}{token_hash}")

        logger.info("Token révoqué: %s...", token_hash[:8])

        # ✅ Métrique Prometheus
        try:
            from security.security_metrics import tokens_revoked_total

            tokens_revoked_total.labels(token_type="refresh_token").inc()
        except Exception:
            pass  # Ne pas bloquer si métriques indisponibles

    def store_token(
        self,
        user_id: int,
        token: str,
        ttl_seconds: int | None = None,
    ) -> None:
        """Stocke un nouveau token actif.

        Args:
            user_id: ID de l'utilisateur propriétaire du token
            token: Le refresh token JWT en clair
            ttl_seconds: Durée de vie spécifique en secondes pour ce token. Si
                ``None``, utilise ``JWT_REFRESH_TOKEN_EXPIRES`` (compatibilité
                rétro). Permet d'aligner le TTL Redis sur l'expiration JWT
                effective (ex: refresh token court pour remember_me=False).
        """
        import time

        token_hash = self._hash_token(token)

        if ttl_seconds is not None and ttl_seconds > 0:
            ttl = int(ttl_seconds)
        else:
            try:
                refresh_expires_delta = current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
                ttl = int(refresh_expires_delta.total_seconds())
            except Exception:
                ttl = 30 * 24 * 3600

        self.redis_client.setex(
            f"{self.active_tokens_prefix}{token_hash}",
            ttl,
            str(user_id),
        )

        # ZSET ordonné par timestamp (score) pour éviction FIFO déterministe
        user_tokens_key = f"user_refresh_tokens:{user_id}"

        def _zadd() -> Any:
            self.redis_client.zadd(user_tokens_key, {token_hash: time.time()})

        _fix_wrongtype_and_retry(self.redis_client, user_tokens_key, _zadd)
        self.redis_client.expire(user_tokens_key, ttl)

        logger.info("Token stocké: user_id=%s, hash=%s...", user_id, token_hash[:8])

    def revoke_all_user_tokens(self, user_id: int) -> None:
        """Révoque tous les tokens actifs d'un utilisateur (pour logout).

        Args:
            user_id: ID de l'utilisateur
        """
        user_tokens_key = f"user_refresh_tokens:{user_id}"

        def _zrange() -> Any:
            return self.redis_client.zrange(user_tokens_key, 0, -1)

        token_hashes = _fix_wrongtype_and_retry(
            self.redis_client, user_tokens_key, _zrange
        )

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

        def _zcard() -> Any:
            return self.redis_client.zcard(user_tokens_key)

        return _fix_wrongtype_and_retry(self.redis_client, user_tokens_key, _zcard)

    def touch_token_score(self, user_id: int, token: str) -> None:
        """Met a jour le score ZSET d'un token avec le timestamp actuel.

        Cela garantit que la purge (limit_active_tokens) evince les tokens
        les moins recemment utilises, pas les plus anciens par creation.
        """
        import time

        token_hash = self._hash_token(token)
        user_tokens_key = f"user_refresh_tokens:{user_id}"

        def _zscore() -> Any:
            return self.redis_client.zscore(user_tokens_key, token_hash)

        def _zadd() -> Any:
            self.redis_client.zadd(user_tokens_key, {token_hash: time.time()})

        if (
            _fix_wrongtype_and_retry(self.redis_client, user_tokens_key, _zscore)
            is not None
        ):
            _fix_wrongtype_and_retry(self.redis_client, user_tokens_key, _zadd)

    def limit_active_tokens(self, user_id: int, max_tokens: int = 5) -> None:
        """Limite le nombre de tokens actifs par utilisateur.

        Evince les tokens les plus anciens (score le plus bas dans le ZSET)
        pour garantir un comportement FIFO deterministe.

        Args:
            user_id: ID de l'utilisateur
            max_tokens: Nombre maximum de tokens actifs (defaut: 5)
        """
        user_tokens_key = f"user_refresh_tokens:{user_id}"

        def _zcard() -> Any:
            return self.redis_client.zcard(user_tokens_key)

        count = _fix_wrongtype_and_retry(self.redis_client, user_tokens_key, _zcard)

        if count > max_tokens:
            excess = count - max_tokens

            def _zrange() -> Any:
                return self.redis_client.zrange(user_tokens_key, 0, excess - 1)

            tokens_to_revoke = _fix_wrongtype_and_retry(
                self.redis_client, user_tokens_key, _zrange
            )

            try:
                refresh_expires_delta = current_app.config["JWT_REFRESH_TOKEN_EXPIRES"]
                ttl = int(refresh_expires_delta.total_seconds())
            except Exception:
                ttl = 30 * 24 * 3600

            pruned_hashes = []
            for th in tokens_to_revoke:
                token_hash = th.decode() if isinstance(th, bytes) else th
                self.redis_client.setex(
                    f"{self.revoked_tokens_prefix}{token_hash}",
                    ttl,
                    "revoked",
                )
                self.redis_client.delete(f"{self.active_tokens_prefix}{token_hash}")
                self.redis_client.zrem(user_tokens_key, token_hash)
                pruned_hashes.append(token_hash[:8])

            logger.info(
                "refresh_token_pruned user_id=%s before=%s after=%s pruned=%s",
                user_id,
                count,
                max_tokens,
                pruned_hashes,
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
