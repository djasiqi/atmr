# shared/avs_utils.py
"""Utilitaires pour la gestion des numéros AVS (assurance vieillesse et survivants).

Le numéro AVS suisse est un identifiant à 13 chiffres (format 756.XXXX.XXXX.XX)
conforme à la norme EAN-13.

Sécurité : l'AVS en clair n'est jamais stocké dans le master index.
On utilise HMAC-SHA256 avec un pepper secret (variable d'env) pour l'indexation.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re

# Pepper secret en variable d'env (jamais en DB, jamais en git).
# Rend le hash AVS résistant aux attaques par rainbow table
# (espace de recherche = 13 chiffres = ~10^13, attaquable sans pepper).
_AVS_PEPPER = os.environ.get("AVS_HASH_PEPPER", "")


def normalize_avs(avs_raw: str) -> str:
    """Normalise un numéro AVS en supprimant tout sauf les chiffres.

    756.1234.5678.97 -> 7561234567897
    """
    return re.sub(r"[^0-9]", "", avs_raw)


def hash_avs(avs_raw: str) -> str:
    """Retourne le HMAC-SHA256 de l'AVS normalisé avec pepper serveur."""
    normalized = normalize_avs(avs_raw)
    return hmac.new(
        _AVS_PEPPER.encode(), normalized.encode(), hashlib.sha256
    ).hexdigest()


def last4_avs(avs_raw: str) -> str:
    """Retourne les 4 derniers chiffres de l'AVS."""
    return normalize_avs(avs_raw)[-4:]


def validate_avs(avs_raw: str) -> str:
    """Valide la structure d'un numéro AVS suisse (EAN-13).

    Returns:
        'valid' — AVS suisse valide (13 chiffres, préfixe 756, check digit OK)
        'invalid' — Structure incorrecte ou check digit erroné
        'unknown' — 13 chiffres mais pas préfixe 756 (format étranger possible)
    """
    digits = normalize_avs(avs_raw)
    if len(digits) != 13:
        return "invalid"
    if not digits.startswith("756"):
        return "unknown"
    # EAN-13 check digit : somme des 12 premiers (poids 1,3 alternée)
    total = sum(
        int(d) * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits[:12])
    )
    expected_check = (10 - total % 10) % 10
    return "valid" if int(digits[12]) == expected_check else "invalid"
