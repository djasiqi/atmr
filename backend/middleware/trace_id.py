"""Middleware pour gérer les trace_id dans les requêtes API.

Génère et injecte des trace_id pour le traçage des requêtes.
"""

import uuid
import logging
from typing import Any

from flask import request, g  # pyright: ignore[reportMissingImports]

logger = logging.getLogger(__name__)


def generate_trace_id() -> str:
    """Génère un trace_id unique.

    Returns:
        Trace ID au format UUID v4 (32 caractères hex)
    """
    return uuid.uuid4().hex


def get_trace_id() -> str:
    """Récupère le trace_id de la requête actuelle.

    Génère un nouveau trace_id si absent.

    Returns:
        Trace ID de la requête
    """
    if not hasattr(g, "trace_id"):
        # Vérifier si un trace_id est fourni dans les headers
        trace_id = request.headers.get("X-Trace-Id") or request.headers.get("Trace-Id")
        if trace_id:
            g.trace_id = trace_id
        else:
            g.trace_id = generate_trace_id()
            logger.debug("Trace ID généré: %s", g.trace_id)

    return g.trace_id


def inject_trace_id_middleware():
    """Middleware Flask pour injecter trace_id dans g.

    À appeler avant chaque requête.
    """
    get_trace_id()  # Génère/injecte trace_id dans g


def add_trace_id_to_response(response: Any) -> Any:
    """Ajoute le trace_id dans les headers de réponse.

    Args:
        response: Réponse Flask

    Returns:
        Réponse avec header X-Trace-Id ajouté
    """
    trace_id = get_trace_id()
    response.headers["X-Trace-Id"] = trace_id
    return response


def get_trace_id_for_logging() -> dict[str, str]:
    """Retourne un dictionnaire avec trace_id pour les logs structurés.

    Returns:
        Dict avec clé 'trace_id'
    """
    return {"trace_id": get_trace_id()}
