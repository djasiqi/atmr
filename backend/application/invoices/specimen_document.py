"""Spécimen de recette : document visible, jamais payable ni émissible."""

from __future__ import annotations

from typing import Any

SPECIMEN_REFUSAL = (
    "Ce document est un spécimen non émis. Aucune émission ni encaissement."
)


def is_non_payable_specimen(document: Any) -> bool:
    """Vrai si le document porte le marquage spécimen ou payable=false."""
    meta = getattr(document, "meta", None)
    if not isinstance(meta, dict):
        return False
    return meta.get("specimen") is True or meta.get("payable") is False
