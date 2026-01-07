"""Décorateurs réutilisables pour l'application.

Ce module contient des décorateurs utilitaires pour la sécurité,
la validation, etc.
"""

import logging
from contextlib import suppress
from functools import wraps

from flask import request  # pyright: ignore[reportMissingImports]

from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)


def csrf_required(f):
    """Décorateur pour vérifier le token CSRF sur les endpoints mutants.

    Ce décorateur doit être utilisé sur les endpoints qui effectuent
    des modifications (POST, PUT, DELETE, PATCH) pour protéger contre
    les attaques CSRF (Cross-Site Request Forgery).

    Le token CSRF doit être envoyé dans le header X-CSRF-Token.

    Usage:
        @app.route("/api/users", methods=["POST"])
        @jwt_required()
        @csrf_required
        def create_user():
            ...

    Args:
        f: Fonction à décorer

    Returns:
        Fonction décorée avec validation CSRF

    Raises:
        ValidationError: Si le token CSRF est manquant ou invalide
    """

    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Vérifier uniquement pour les méthodes mutantes
        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            # Récupérer le token depuis le header
            csrf_token = request.headers.get("X-CSRF-Token")

            if not csrf_token:
                logger.warning(
                    "⚠️ Requête sans token CSRF: %s %s",
                    request.method,
                    request.path,
                )
                return APIErrorHandler.handle_validation_error(
                    "Token CSRF manquant. Incluez X-CSRF-Token dans les headers.",
                    logger_instance=logger,
                )

            # Vérifier le token (utiliser le service CSRF existant)
            from services.security.csrf import validate_csrf_token

            # Récupérer user_id depuis JWT si disponible
            user_id = None
            try:
                from flask_jwt_extended import get_jwt_identity

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

            if not validate_csrf_token(csrf_token, user_id=user_id):
                logger.warning(
                    "⚠️ Token CSRF invalide: %s %s",
                    request.method,
                    request.path,
                )
                return APIErrorHandler.handle_validation_error(
                    "Token CSRF invalide ou expiré.",
                    logger_instance=logger,
                )

        return f(*args, **kwargs)

    return decorated_function
