"""Middleware pour gérer les trace_id dans les requêtes API.

Génère et injecte des trace_id pour le traçage des requêtes.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from flask import g, request

logger = logging.getLogger(__name__)

# Corrélation uniquement : hex, mobile `mob_*`, tests `e2e-trace-*`.
# Rejette HTML, quotes, CRLF et tout caractère hors allowlist.
_SAFE_CLIENT_TRACE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def generate_trace_id() -> str:
    """Génère un trace_id unique.

    Returns:
        Trace ID au format UUID v4 (32 caractères hex)
    """
    return uuid.uuid4().hex


def sanitize_client_trace_id(raw: str | None) -> str | None:
    """Retourne l'identifiant client s'il est sûr, sinon None."""
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value:
        return None
    if "\r" in value or "\n" in value or "\x00" in value:
        return None
    if not _SAFE_CLIENT_TRACE_ID_RE.fullmatch(value):
        return None
    return value


def get_trace_id() -> str:
    """Récupère le trace_id de la requête actuelle.

    Génère un nouveau trace_id si absent ou si le header client est invalide.

    Returns:
        Trace ID de la requête
    """
    if not hasattr(g, "trace_id"):
        incoming = request.headers.get("X-Trace-Id") or request.headers.get("Trace-Id")
        sanitized = sanitize_client_trace_id(incoming)
        if sanitized:
            g.trace_id = sanitized
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
