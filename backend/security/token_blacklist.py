"""✅ Phase 3: Gestion de la blacklist des tokens JWT.

Utilise Redis pour stocker les tokens révoqués avec expiration automatique (TTL).
"""

import logging
from datetime import UTC, datetime

from flask_jwt_extended import decode_token, get_jwt

from ext import redis_client

logger = logging.getLogger(__name__)

# Préfixe pour les clés Redis
BLACKLIST_PREFIX = "jwt:blacklist:"


def add_to_blacklist(jwt_token: str, ttl_seconds: int | None = None) -> bool:
    """Ajoute un token à la blacklist.

    Args:
        jwt_token: Token JWT à révoquer
        ttl_seconds: TTL en secondes (None = utiliser expiration du token)

    Returns:
        True si ajouté avec succès, False sinon
    """
    if not redis_client:
        logger.warning("[Token Blacklist] Redis non disponible, blacklist désactivée")
        return False

    try:
        # Décoder le token pour obtenir son expiration
        decoded = decode_token(jwt_token)
        exp = decoded.get("exp")

        if exp:
            # Calculer TTL basé sur l'expiration du token
            exp_datetime = datetime.fromtimestamp(exp, tz=UTC)
            now = datetime.now(UTC)
            ttl = int((exp_datetime - now).total_seconds())

            # Si le token est déjà expiré, ne pas l'ajouter
            if ttl <= 0:
                logger.debug(
                    "[Token Blacklist] Token déjà expiré, pas besoin de blacklist"
                )
                return False

            # Utiliser le TTL fourni ou celui du token
            if ttl_seconds is not None:
                ttl = min(ttl, ttl_seconds)
        else:
            # Pas d'expiration dans le token, utiliser TTL fourni ou 24h par défaut
            ttl = ttl_seconds or (24 * 3600)

        # Créer une clé unique pour ce token (jti si disponible, sinon hash du token)
        jti = decoded.get("jti")
        if jti:
            key = f"{BLACKLIST_PREFIX}{jti}"
        else:
            # Fallback: utiliser un hash du token
            import hashlib

            token_hash = hashlib.sha256(jwt_token.encode()).hexdigest()
            key = f"{BLACKLIST_PREFIX}{token_hash}"

        # Stocker dans Redis avec TTL
        redis_client.setex(key, ttl, "1")
        logger.info("[Token Blacklist] Token ajouté à la blacklist (TTL: %d s)", ttl)
        return True

    except Exception as e:
        logger.exception(
            "[Token Blacklist] Erreur lors de l'ajout à la blacklist: %s", e
        )
        return False


def is_token_blacklisted(jwt_token: str | None = None, jti: str | None = None) -> bool:
    """Vérifie si un token est dans la blacklist.

    Args:
        jwt_token: Token JWT complet (optionnel si jti fourni)
        jti: JWT ID (optionnel si jwt_token fourni)

    Returns:
        True si le token est blacklisté, False sinon
    """
    if not redis_client:
        # Si Redis n'est pas disponible, on considère que la blacklist est vide
        return False

    try:
        # Utiliser jti si fourni, sinon extraire du token
        if jti:
            key = f"{BLACKLIST_PREFIX}{jti}"
        elif jwt_token:
            decoded = decode_token(jwt_token)
            jti = decoded.get("jti")
            if jti:
                key = f"{BLACKLIST_PREFIX}{jti}"
            else:
                # Fallback: hash du token
                import hashlib

                token_hash = hashlib.sha256(jwt_token.encode()).hexdigest()
                key = f"{BLACKLIST_PREFIX}{token_hash}"
        else:
            return False

        # Vérifier si la clé existe dans Redis
        exists = redis_client.exists(key)
        return bool(exists)

    except Exception as e:
        logger.exception("[Token Blacklist] Erreur lors de la vérification: %s", e)
        # En cas d'erreur, on considère que le token n'est pas blacklisté (fail-open)
        return False


def revoke_token() -> bool:
    """Révoque le token JWT actuel (depuis le contexte Flask-JWT).

    Returns:
        True si révoqué avec succès, False sinon
    """
    try:
        jwt_data = get_jwt()
        jti = jwt_data.get("jti")

        if not jti:
            logger.warning(
                "[Token Blacklist] Pas de jti dans le token, impossible de révoquer"
            )
            return False

        # Ajouter à la blacklist
        key = f"{BLACKLIST_PREFIX}{jti}"

        # Obtenir l'expiration du token
        exp = jwt_data.get("exp")
        if exp:
            exp_datetime = datetime.fromtimestamp(exp, tz=UTC)
            now = datetime.now(UTC)
            ttl = int((exp_datetime - now).total_seconds())

            if ttl > 0:
                redis_client.setex(key, ttl, "1")
                logger.info(
                    "[Token Blacklist] Token révoqué (jti: %s, TTL: %d s)", jti, ttl
                )
                return True

        return False

    except Exception as e:
        logger.exception("[Token Blacklist] Erreur lors de la révocation: %s", e)
        return False
