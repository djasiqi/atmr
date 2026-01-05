# backend/services/unified_dispatch/orchestration/utils.py
"""Utilitaires partagés pour l'orchestration du dispatch.

Ce module contient des fonctions utilitaires utilisées par plusieurs modules
de l'orchestration du dispatch. Ces fonctions sont extraites de
`dispatch_orchestrator.py` pour améliorer la réutilisabilité et la maintenabilité.

Fonctions:
    to_date_ymd: Conversion de chaîne de date en objet date
    safe_int: Conversion sécurisée en entier
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)


def to_date_ymd(s: str) -> date:
    """Convertit une chaîne en date.

    Accepte 'YYYY-MM-DD' et ISO full (on ne garde que la date).

    Args:
        s: Chaîne de date au format 'YYYY-MM-DD' ou ISO complet

    Returns:
        Date object

    Raises:
        ValueError: Si la chaîne ne peut pas être parsée en date valide

    Exemple:
        >>> to_date_ymd("2025-01-14")
        datetime.date(2025, 1, 14)
        >>> to_date_ymd("2025-01-14T10:30:00")
        datetime.date(2025, 1, 14)
    """
    DATE_FORMAT_LENGTH = 10  # Longueur du format YYYY-MM-DD
    try:
        if len(s) == DATE_FORMAT_LENGTH and s[4] == "-" and s[7] == "-":
            return date.fromisoformat(s)
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError) as err:
        # Erreurs de parsing de date attendues
        msg = f"for_date invalide: {s!r} (attendu 'YYYY-MM-DD')"
        raise ValueError(msg) from err
    except Exception as err:
        # Erreur inattendue : logger et re-lever avec contexte
        logger.exception("Erreur inattendue lors de la conversion de date: %s", s)
        raise ValueError(f"for_date invalide: {s!r}") from err


def safe_int(v: Any) -> int | None:
    """Convertit n'importe quelle valeur en int Python ou retourne None.

    Cette fonction gère gracieusement les erreurs de conversion et retourne
    None au lieu de lever une exception. Utile pour convertir des valeurs
    provenant de l'API ou de la base de données qui peuvent être None ou
    de type inattendu.

    Args:
        v: Valeur à convertir (peut être int, str, float, None, etc.)

    Returns:
        int si la conversion réussit, None sinon

    Exemple:
        >>> safe_int("42")
        42
        >>> safe_int(42.5)
        42
        >>> safe_int("invalid")
        None
        >>> safe_int(None)
        None
    """
    try:
        return int(v)
    except (ValueError, TypeError, OverflowError):
        # Erreurs de conversion attendues : valeur invalide, type incorrect, overflow
        return None
    except Exception:
        # Erreur inattendue : logger et retourner None
        logger.debug("Unexpected error converting to int: %s, returning None", v)
        return None
