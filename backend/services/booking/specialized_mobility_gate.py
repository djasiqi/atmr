"""Garde : prestations spécialisées non ouvertes (brancard / oxygène).

Par défaut refusées. N'active pas le produit — empêche l'acceptation silencieuse.
"""

from __future__ import annotations

import os
from typing import Any, Final, Mapping

SPECIALIZED_MOBILITY_ENABLED: Final[bool] = os.getenv(
    "SPECIALIZED_MOBILITY_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}

_BLOCKED_KEYS: Final[tuple[tuple[str, str], ...]] = (
    ("stretcher", "brancard"),
    ("oxygen", "oxygène"),
)


def blocked_specialized_mobility_labels(
    mobility: Mapping[str, Any] | None,
) -> list[str]:
    """Retourne les libellés des besoins spécialisés présents et non autorisés."""
    if SPECIALIZED_MOBILITY_ENABLED or not mobility:
        return []
    labels: list[str] = []
    for key, label in _BLOCKED_KEYS:
        if mobility.get(key):
            labels.append(label)
    return labels


def reject_if_specialized_mobility_blocked(
    mobility: Mapping[str, Any] | None,
) -> str | None:
    """Message d'erreur FR si un besoin spécialisé est demandé alors que gelé."""
    labels = blocked_specialized_mobility_labels(mobility)
    if not labels:
        return None
    joined = ", ".join(labels)
    return (
        f"Prestation non disponible actuellement ({joined}). "
        "Contactez votre transporteur pour une solution adaptée."
    )
