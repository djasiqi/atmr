"""Normalisation des libellés transport dupliqués (factures PDF + JSON).

- **RIDE** : uniquement les motifs « Trajet : Trajet … » (pas de règles Livraison).
- **MATERIAL_DELIVERY** : uniquement les motifs « Livraison – Livraison … » (pas de Trajet).
"""

from __future__ import annotations

import re
from typing import Literal

TransportLineKind = Literal["ride", "material_delivery"]

# Répétitions « Trajet : Trajet : … Trajet » avant le libellé réel (ex. rue).
_TRAJET_DUP = re.compile(
    r"Trajet\s*[:：\uff1a]\s*(?:Trajet\s*[:：\uff1a]\s*)*Trajet(?:\s+|(?=[A-Za-zÀ-ÿ0-9])|$)",
    re.IGNORECASE,
)
_LIVRAISON_DASH_DUP = re.compile(
    r"Livraison\s*[-–—]\s*(?:Livraison\s*[-–—]\s*)*Livraison(?:\s+|(?=[A-Za-zÀ-ÿ0-9])|$)",
    re.IGNORECASE,
)
_LIVRAISON_COLON_DUP = re.compile(
    r"Livraison\s*[:：\uff1a]\s*(?:Livraison\s*[:：\uff1a]\s*)*Livraison(?:\s+|(?=[A-Za-zÀ-ÿ0-9])|$)",
    re.IGNORECASE,
)


def _clean_entities(text: str) -> str:
    return (
        str(text)
        .replace("&nbsp;", " ")
        .replace("&#160;", " ")
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
        .replace("\u200b", "")
        .replace("\u200c", "")
        .replace("\u200d", "")
        .replace("\ufeff", "")
        .replace("\u2060", "")
    )


def normalize_transport_line_description(
    text: str | None,
    *,
    kind: TransportLineKind,
) -> str:
    """Supprime les doublons de libellé selon le type de ligne (trajet **ou** livraison, pas les deux).

    Idempotent.
    """
    if text is None:
        return ""
    t = _clean_entities(text).strip()
    prev = None
    while prev != t:
        prev = t
        if kind == "ride":
            t = _TRAJET_DUP.sub("Trajet : ", t)
        else:
            t = _LIVRAISON_DASH_DUP.sub("Livraison – ", t)
            t = _LIVRAISON_COLON_DUP.sub("Livraison : ", t)
    return t.strip()
