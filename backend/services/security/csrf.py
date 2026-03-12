"""Protection CSRF (Cross-Site Request Forgery) pour l'application ATMR.

Ce module fournit :
1. Génération de tokens CSRF
2. Validation de tokens CSRF
3. Middleware pour protection automatique des routes mutantes
"""

import hashlib
import hmac
import logging
import os
import secrets
import time
from contextlib import suppress
from typing import Any

from flask import request

logger = logging.getLogger(__name__)

# Durée de vie du token CSRF (1 heure par défaut)
CSRF_TOKEN_TTL = int(os.getenv("CSRF_TOKEN_TTL_SECONDS", "3600"))

# Header personnalisé pour le token CSRF (alternative à X-CSRF-Token)
CSRF_HEADER_NAME = "X-CSRF-Token"
CSRF_COOKIE_NAME = "csrf_token"  # Pour support cookie (optionnel)


def _is_production() -> bool:
    """Détecte si l'application est en mode production."""
    try:
        from flask import current_app

        if current_app and current_app.config:
            env = (
                current_app.config.get("ENVIRONMENT")
                or current_app.config.get("FLASK_CONFIG")
                or ""
            )
            if isinstance(env, str) and env.lower() == "production":
                return True
    except Exception:
        pass
    return (
        os.getenv("FLASK_CONFIG", "").lower() == "production"
        or os.getenv("FLASK_ENV", "").lower() == "production"
    )


def _get_csrf_secret() -> str:
    """Récupère la clé secrète pour signer les tokens CSRF.

    Utilise JWT_SECRET_KEY ou SECRET_KEY ou FLASK_SECRET_KEY.
    En production : aucun fallback autorisé (RuntimeError si secret manquant).
    En dev/test : fallback toléré avec warning.
    """
    secret = (
        os.getenv("JWT_SECRET_KEY")
        or os.getenv("SECRET_KEY")
        or os.getenv("FLASK_SECRET_KEY")
    )
    if not secret:
        if _is_production():
            raise RuntimeError(
                "[CSRF] Production: aucune clé secrète trouvée (JWT_SECRET_KEY, "
                "SECRET_KEY ou FLASK_SECRET_KEY). Configurez une clé valide."
            )
        logger.warning(
            "[CSRF] ⚠️ Aucune clé secrète trouvée pour CSRF. "
            "Utilisation d'une clé temporaire (dev/test uniquement)."
        )
        secret = "temporary-csrf-secret-change-in-production"
    return secret


def generate_csrf_token(
    user_id: int | None = None, session_id: str | None = None
) -> str:
    """Génère un token CSRF signé.

    Le token est signé avec HMAC-SHA256 pour garantir son authenticité.

    Args:
        user_id: ID de l'utilisateur (optionnel, pour lier le token à un utilisateur)
        session_id: ID de session (optionnel, pour lier le token à une session)

    Returns:
        Token CSRF (format: timestamp:nonce:signature)
    """
    secret = _get_csrf_secret()
    timestamp = str(int(time.time()))
    nonce = secrets.token_urlsafe(16)

    # Construire le message à signer
    message_parts = [timestamp, nonce]
    if user_id is not None:
        message_parts.append(str(user_id))
    if session_id:
        message_parts.append(session_id)

    message = ":".join(message_parts)

    # Signer avec HMAC-SHA256
    signature = hmac.new(
        secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
    ).hexdigest()

    # Token format: timestamp:nonce:signature
    token = f"{timestamp}:{nonce}:{signature}"

    logger.debug("[CSRF] Token généré pour user_id=%s", user_id)
    return token


def validate_csrf_token(
    token: str, user_id: int | None = None, session_id: str | None = None
) -> bool:
    """Valide un token CSRF.

    Args:
        token: Token CSRF à valider
        user_id: ID de l'utilisateur (optionnel, pour vérifier la correspondance)
        session_id: ID de session (optionnel, pour vérifier la correspondance)

    Returns:
        True si le token est valide, False sinon
    """
    if not token:
        logger.warning("[CSRF] Token manquant")
        # ✅ PHASE 3: Métrique Prometheus
        try:
            from security.security_metrics import csrf_validation_failures_total

            csrf_validation_failures_total.inc()
        except Exception:
            pass  # Ne pas bloquer si métriques indisponibles
        return False

    try:
        # Parser le token (format: timestamp:nonce:signature)
        TOKEN_PARTS_COUNT = 3
        parts = token.split(":")
        if len(parts) != TOKEN_PARTS_COUNT:
            logger.warning("[CSRF] Format de token invalide")
            # ✅ PHASE 3: Métrique Prometheus
            try:
                from security.security_metrics import csrf_validation_failures_total

                csrf_validation_failures_total.inc()
            except Exception:
                pass  # Ne pas bloquer si métriques indisponibles
            return False

        timestamp_str, nonce, signature = parts
        timestamp = int(timestamp_str)

        # Vérifier l'expiration
        current_time = int(time.time())
        if current_time - timestamp > CSRF_TOKEN_TTL:
            logger.warning(
                "[CSRF] Token expiré (âge: %ds, TTL: %ds)",
                current_time - timestamp,
                CSRF_TOKEN_TTL,
            )
            # ✅ PHASE 3: Métrique Prometheus
            try:
                from security.security_metrics import csrf_validation_failures_total

                csrf_validation_failures_total.inc()
            except Exception:
                pass  # Ne pas bloquer si métriques indisponibles
            return False

        # Reconstruire le message
        message_parts = [timestamp_str, nonce]
        if user_id is not None:
            message_parts.append(str(user_id))
        if session_id:
            message_parts.append(session_id)

        message = ":".join(message_parts)

        # Vérifier la signature
        secret = _get_csrf_secret()
        expected_signature = hmac.new(
            secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        # Comparaison sécurisée (timing-safe)
        if not hmac.compare_digest(signature, expected_signature):
            logger.warning("[CSRF] Signature invalide")
            # ✅ PHASE 3: Métrique Prometheus
            try:
                from security.security_metrics import csrf_validation_failures_total

                csrf_validation_failures_total.inc()
            except Exception:
                pass  # Ne pas bloquer si métriques indisponibles
            return False

        logger.debug("[CSRF] Token valide pour user_id=%s", user_id)
        return True

    except (ValueError, TypeError) as e:
        logger.warning("[CSRF] Erreur lors de la validation: %s", e)
        # ✅ PHASE 3: Métrique Prometheus
        try:
            from security.security_metrics import csrf_validation_failures_total

            csrf_validation_failures_total.inc()
        except Exception:
            pass  # Ne pas bloquer si métriques indisponibles
        return False


def get_csrf_token_from_request() -> str | None:
    """Récupère le token CSRF depuis la requête HTTP.

    Cherche le token dans :
    1. Header `X-CSRF-Token`
    2. Header `X-Csrf-Token` (variante)
    3. Cookie `csrf_token` (si supporté)
    4. Body JSON `csrf_token` (pour POST avec JSON)

    Returns:
        Token CSRF ou None si non trouvé
    """
    # 1. Header X-CSRF-Token (priorité)
    token = request.headers.get(CSRF_HEADER_NAME) or request.headers.get("X-Csrf-Token")
    if token:
        return token

    # 2. Cookie csrf_token (si supporté)
    token = request.cookies.get(CSRF_COOKIE_NAME)
    if token:
        return token

    # 3. Body JSON csrf_token (pour POST avec JSON)
    if request.is_json:
        try:
            data = request.get_json(silent=True)
            if data and isinstance(data, dict):
                token = data.get("csrf_token") or data.get("csrfToken")
                if token:
                    return token
        except Exception:
            pass

    return None


def require_csrf_token(f: Any) -> Any:
    """Décorateur pour exiger un token CSRF valide sur une route.

    Usage:
        @app.route("/api/endpoint", methods=["POST"])
        @require_csrf_token
        def my_endpoint():
            ...

    Args:
        f: Fonction à décorer

    Returns:
        Fonction décorée avec validation CSRF
    """
    from functools import wraps

    @wraps(f)
    def decorated_function(*args: Any, **kwargs: Any) -> Any:
        # Récupérer le token depuis la requête
        token = get_csrf_token_from_request()

        if not token:
            logger.warning(
                "[CSRF] Token manquant pour %s %s",
                request.method,
                request.path,
            )
            return {"error": "Token CSRF manquant"}, 403

        # Récupérer user_id depuis JWT (si disponible)
        user_id = None
        try:
            from flask_jwt_extended import (
                get_jwt_identity,
            )

            jwt_identity = get_jwt_identity()
            if jwt_identity:
                # JWT identity peut être un dict ou un int
                if isinstance(jwt_identity, dict):
                    user_id = jwt_identity.get("user_id") or jwt_identity.get("id")
                elif isinstance(jwt_identity, (int, str)):
                    with suppress(ValueError, TypeError):
                        user_id = int(jwt_identity)
        except Exception:
            # JWT non disponible ou non configuré, continuer sans user_id
            pass

        # Valider le token
        if not validate_csrf_token(token, user_id=user_id):
            logger.warning(
                "[CSRF] Token invalide pour %s %s (user_id=%s)",
                request.method,
                request.path,
                user_id,
            )
            return {"error": "Token CSRF invalide ou expiré"}, 403

        # Token valide, continuer
        return f(*args, **kwargs)

    return decorated_function


def setup_csrf_protection(app: Any) -> None:
    """Configure la protection CSRF globale pour l'application.

    Ajoute un middleware qui valide automatiquement les tokens CSRF
    pour toutes les requêtes mutantes (POST/PUT/PATCH/DELETE).

    Args:
        app: Instance Flask
    """
    # Liste des endpoints exemptés de CSRF (ex: endpoints publics, webhooks)
    csrf_exempt_paths = {
        "/health",
        "/api/v1/prometheus/metrics",
        "/api/v1/auth/login",
        "/api/v1/auth/register",
        "/api/v1/auth/refresh-token",
        "/api/v1/auth/login-test",
        "/api/auth/login-test",
        "/api/v1/csrf-token",
        "/api/gateway/auth/login",
        "/api/v1/contact/requests",
        "/api/contact/requests",
        "/api/app/contact/requests",
        "/api/demo/contact/requests",
        "/api/v1/demo-requests",
        "/api/demo-requests",
        "/api/app/demo-requests",
        "/api/demo/demo-requests",
        "/api/v1/demo_access/consume-magic-link",
        "/api/demo_access/consume-magic-link",
        "/api/app/demo_access/consume-magic-link",
        "/api/demo/demo_access/consume-magic-link",
        "/api/v1/demo_access/analytics",
        "/api/demo_access/analytics",
        "/api/app/demo_access/analytics",
        "/api/demo/demo_access/analytics",
        "/api/v1/app/version-check",
        "/api/v1/company_mobile/auth/login",
    }

    @app.before_request
    def validate_csrf_for_mutating_requests():  # pyright: ignore[reportUnusedFunction]
        """Middleware pour valider CSRF sur les requêtes mutantes."""
        should_check = True

        # ✅ Tests: désactiver CSRF si l'app est en mode TESTING ou si CSRF est désactivé
        try:
            from flask import current_app

            if current_app and (
                bool(current_app.config.get("TESTING", False))
                or not bool(current_app.config.get("CSRF_ENABLED", True))
                or not bool(current_app.config.get("WTF_CSRF_ENABLED", True))
            ):
                should_check = False
        except Exception:
            # Défensif: ne pas bloquer si current_app n'est pas disponible
            pass

        # Ignorer les méthodes non mutantes
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            should_check = False

        # Ignorer les endpoints exemptés
        if request.path in csrf_exempt_paths:
            should_check = False

        csrf_exempt_prefixes = (
            "/api/v1/auth/",
            "/api/gateway/auth/",
            "/api/v1/webhooks/",
            "/api/v1/company_mobile/",
            "/api/v1/driver/",
            "/api/v1/companies/",
            "/api/v1/company_dispatch/",
            "/api/v1/dispatch/",
            "/api/dispatch/",
            "/api/v1/demo-requests",
            "/api/demo-requests",
            "/api/app/demo-requests",
            "/api/demo/demo-requests",
            "/api/v1/demo_access/",
            "/api/demo_access/",
            "/api/app/demo_access/",
            "/api/demo/demo_access/",
        )
        if request.path.startswith(csrf_exempt_prefixes):
            should_check = False

        if not should_check:
            return None

        # Récupérer le token depuis la requête
        token = get_csrf_token_from_request()
        if not token:
            logger.warning(
                "[CSRF] Token manquant pour %s %s",
                request.method,
                request.path,
            )
            # ✅ PHASE 3: Métrique Prometheus
            try:
                from security.security_metrics import csrf_validation_failures_total

                csrf_validation_failures_total.inc()
            except Exception:
                pass  # Ne pas bloquer si métriques indisponibles
            return {"error": "Token CSRF manquant"}, 403

        # Récupérer user_id depuis JWT (si disponible)
        user_id = None
        try:
            from flask_jwt_extended import (
                get_jwt_identity,
            )

            jwt_identity = get_jwt_identity()
            if jwt_identity:
                if isinstance(jwt_identity, dict):
                    user_id = jwt_identity.get("user_id") or jwt_identity.get("id")
                elif isinstance(jwt_identity, (int, str)):
                    with suppress(ValueError, TypeError):
                        user_id = int(jwt_identity)
        except Exception:
            # JWT non disponible, continuer sans user_id
            pass

        # Valider le token
        if not validate_csrf_token(token, user_id=user_id):
            logger.warning(
                "[CSRF] Token invalide pour %s %s (user_id=%s)",
                request.method,
                request.path,
                user_id,
            )
            return {"error": "Token CSRF invalide ou expiré"}, 403

        return None

    logger.info("[CSRF] ✅ Protection CSRF activée pour requêtes mutantes")
