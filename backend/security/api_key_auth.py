# security/api_key_auth.py
# pyright: reportArgumentType=false, reportUnnecessaryComparison=false
"""Authentification et autorisation par API Key pour DPI.

Fournit:
- Middleware pour authentification via header X-API-Key
- Décorateur @api_key_required(scopes=[...]) pour protéger les routes
- Rate limiting par institution
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from flask import abort, g, request

from ext import db, redis_client

if TYPE_CHECKING:
    from models.institution_api_key import InstitutionApiKey

logger = logging.getLogger(__name__)

# Configuration rate limiting
API_KEY_RATE_LIMIT_PER_MINUTE = int(os.getenv("API_KEY_RATE_LIMIT_PER_MINUTE", "60"))
API_KEY_RATE_LIMIT_WINDOW_SECONDS = 60

# Throttle pour last_used_at (éviter update à chaque requête)
LAST_USED_UPDATE_INTERVAL_SECONDS = int(
    os.getenv("API_KEY_LAST_USED_UPDATE_INTERVAL", "60")
)

# Constante pour le log de prefix de clé API
API_KEY_PREFIX_LOG_LENGTH = 12

F = TypeVar("F", bound=Callable[..., Any])


def get_api_key_from_request() -> str | None:
    """Extrait la clé API du header X-API-Key.

    Returns:
        La clé API ou None si absente
    """
    return request.headers.get("X-API-Key")


def authenticate_api_key() -> InstitutionApiKey | None:
    """Authentifie une requête via X-API-Key.

    Returns:
        InstitutionApiKey si authentification réussie, None sinon.
        Injecte dans g: institution_api_key, institution_id, scopes
    """
    from models.institution_api_key import InstitutionApiKey

    raw_key = get_api_key_from_request()
    if not raw_key:
        return None

    # Rechercher la clé active
    api_key = InstitutionApiKey.find_active_by_raw_key(raw_key)
    if not api_key:
        logger.warning(
            "[API Key Auth] Clé invalide ou révoquée: prefix=%s",
            raw_key[:API_KEY_PREFIX_LOG_LENGTH]
            if len(raw_key) > API_KEY_PREFIX_LOG_LENGTH
            else raw_key,
        )
        return None

    # Injecter dans le contexte Flask
    g.institution_api_key = api_key
    g.institution_id = api_key.institution_id
    g.scopes = set(api_key.get_scopes())
    g.auth_method = "api_key"

    # Mettre à jour last_used_at (avec throttling)
    _update_last_used_throttled(api_key)

    logger.debug(
        "[API Key Auth] Authentification réussie: key_id=%s, institution_id=%s",
        api_key.id,
        api_key.institution_id,
    )

    return api_key


def _update_last_used_throttled(api_key: InstitutionApiKey) -> None:
    """Met à jour last_used_at avec throttling pour éviter trop d'écritures DB.

    Args:
        api_key: La clé API à mettre à jour
    """
    now = datetime.now(UTC)

    # Vérifier si mise à jour nécessaire (throttling)
    if api_key.last_used_at is not None:
        elapsed = (now - api_key.last_used_at).total_seconds()
        if elapsed < LAST_USED_UPDATE_INTERVAL_SECONDS:
            return  # Pas besoin de mettre à jour

    try:
        api_key.last_used_at = now
        db.session.commit()
    except Exception as e:
        logger.warning("[API Key Auth] Échec mise à jour last_used_at: %s", e)
        db.session.rollback()


def check_rate_limit(institution_id: int) -> tuple[bool, int]:
    """Vérifie le rate limit pour une institution.

    Args:
        institution_id: ID de l'institution

    Returns:
        Tuple (is_allowed, remaining):
        - is_allowed: True si la requête est autorisée
        - remaining: Nombre de requêtes restantes
    """
    if not redis_client:
        # Redis non disponible, autoriser par défaut
        logger.debug("[API Key Rate Limit] Redis non disponible, skip rate limit")
        return True, API_KEY_RATE_LIMIT_PER_MINUTE

    key = f"api_key_rate_limit:institution:{institution_id}"

    try:
        # Incrémenter le compteur
        current: int = int(redis_client.incr(key))

        # Définir l'expiration si première requête
        if current == 1:
            redis_client.expire(key, API_KEY_RATE_LIMIT_WINDOW_SECONDS)

        remaining = max(0, API_KEY_RATE_LIMIT_PER_MINUTE - current)
        is_allowed = current <= API_KEY_RATE_LIMIT_PER_MINUTE

        if not is_allowed:
            logger.warning(
                "[API Key Rate Limit] Limite dépassée: institution_id=%s, count=%s",
                institution_id,
                current,
            )

        return is_allowed, remaining

    except Exception as e:
        logger.warning("[API Key Rate Limit] Erreur Redis: %s", e)
        # En cas d'erreur, autoriser par défaut
        return True, API_KEY_RATE_LIMIT_PER_MINUTE


def require_scope(scope: str) -> None:
    """Vérifie qu'un scope est présent dans le contexte.

    Args:
        scope: Scope requis (ex: "requests:write")

    Raises:
        HTTPException 403 si le scope est manquant
    """
    scopes = getattr(g, "scopes", set())
    if scope not in scopes:
        logger.warning(
            "[API Key Auth] Scope manquant: required=%s, available=%s",
            scope,
            scopes,
        )
        abort(403, description=f"Scope '{scope}' required")


def api_key_required(scopes: list[str] | None = None) -> Callable[[F], F]:
    """Décorateur pour protéger une route avec authentification API Key.

    Args:
        scopes: Liste des scopes requis (optionnel)

    Returns:
        Décorateur

    Usage:
        @api_key_required(scopes=["requests:read"])
        def my_endpoint():
            ...
    """

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Authentifier via API Key
            api_key = authenticate_api_key()

            if not api_key:
                logger.warning(
                    "[API Key Auth] Authentification échouée: endpoint=%s",
                    request.path,
                )
                abort(401, description="Invalid or missing API key")

            # Vérifier rate limit
            is_allowed, remaining = check_rate_limit(api_key.institution_id)
            if not is_allowed:
                abort(429, description="Rate limit exceeded")

            # Ajouter header rate limit dans la réponse
            g.rate_limit_remaining = remaining

            # Vérifier scopes si spécifiés
            if scopes:
                for scope in scopes:
                    require_scope(scope)

            return fn(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def api_key_or_jwt_required(scopes: list[str] | None = None) -> Callable[[F], F]:
    """Décorateur pour permettre auth par API Key OU JWT.

    Utile pour les endpoints qui doivent être accessibles par DPI et UI.

    Args:
        scopes: Liste des scopes requis (pour API Key uniquement)

    Returns:
        Décorateur
    """
    from flask_jwt_extended import verify_jwt_in_request

    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Essayer d'abord l'API Key
            api_key = authenticate_api_key()

            if api_key:
                # API Key auth réussie
                # Vérifier rate limit
                is_allowed, remaining = check_rate_limit(api_key.institution_id)
                if not is_allowed:
                    abort(429, description="Rate limit exceeded")

                g.rate_limit_remaining = remaining

                # Vérifier scopes
                if scopes:
                    for scope in scopes:
                        require_scope(scope)

                return fn(*args, **kwargs)

            # Fallback sur JWT
            try:
                verify_jwt_in_request()
                g.auth_method = "jwt"
                return fn(*args, **kwargs)
            except Exception:
                # Ni API Key ni JWT valide
                abort(401, description="Authentication required")

        return wrapper  # type: ignore

    return decorator


# Middleware pour ajouter les headers rate limit dans la réponse
def add_rate_limit_headers(response):
    """Ajoute les headers rate limit à la réponse si applicable."""
    if hasattr(g, "rate_limit_remaining"):
        response.headers["X-RateLimit-Remaining"] = str(g.rate_limit_remaining)
        response.headers["X-RateLimit-Limit"] = str(API_KEY_RATE_LIMIT_PER_MINUTE)
    return response
