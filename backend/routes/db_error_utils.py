"""Convertit les IntegrityError en messages API contrôlés.

Aucun détail SQL / nom de contrainte / requête n'est renvoyé au client.
"""

from typing import Any, Dict, Tuple

from sqlalchemy.exc import IntegrityError

_PGCODE_RESPONSES: dict[str, tuple[dict[str, Any], int]] = {
    "23503": (
        {
            "error": "foreign_key_violation",
            "message": "Référence invalide dans les données.",
        },
        400,
    ),
    "23505": (
        {
            "error": "unique_constraint_violation",
            "message": "Cette valeur existe déjà.",
        },
        400,
    ),
    "23514": (
        {
            "error": "check_constraint_violation",
            "message": (
                "Valeur invalide pour ce champ. Vérifiez les contraintes de validation."
            ),
        },
        400,
    ),
    "23502": (
        {
            "error": "not_null_violation",
            "message": "Un champ obligatoire est manquant.",
        },
        400,
    ),
}


def format_integrity_error(error: IntegrityError) -> Tuple[Dict[str, Any], int]:
    """Mappe un code PostgreSQL vers un message métier stable."""
    error_code = None
    orig = getattr(error, "orig", None)
    if orig is not None:
        error_code = getattr(orig, "pgcode", None)

    if error_code in _PGCODE_RESPONSES:
        body, status = _PGCODE_RESPONSES[error_code]
        return dict(body), status

    return {
        "error": "database_constraint_error",
        "message": "Erreur de contrainte de base de données. Vérifiez vos données.",
    }, 400
