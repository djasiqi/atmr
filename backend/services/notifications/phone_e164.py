"""Normalisation et validation des numéros de téléphone (E.164)."""

from __future__ import annotations

import re

_MIN_E164_DIGITS = 8
_MAX_E164_DIGITS = 15
_SWISS_NATIONAL_LEN = 10
_SWISS_E164_DIGITS = 11


def normalize_e164_phone(phone: str | None) -> str | None:
    """Normalise un numéro suisse ou international vers E.164.

    Accepte notamment :
    - ``+41768190077``
    - ``0041768190077``
    - ``0768190077`` / ``076 819 00 77``
    - ``41768190077``

    Returns:
        Numéro E.164 (``+…``) ou ``None`` si le format n'est pas acceptable.
    """
    raw = (phone or "").strip()
    if not raw:
        return None

    digits = "".join(ch for ch in raw if ch.isdigit())
    if not digits:
        return None

    if raw.startswith("+"):
        if _MIN_E164_DIGITS <= len(digits) <= _MAX_E164_DIGITS:
            return f"+{digits}"
        return None

    if digits.startswith("00"):
        international = digits[2:]
        if _MIN_E164_DIGITS <= len(international) <= _MAX_E164_DIGITS:
            return f"+{international}"
        return None

    if digits.startswith("0") and len(digits) == _SWISS_NATIONAL_LEN:
        return f"+41{digits[1:]}"

    if digits.startswith("41") and len(digits) == _SWISS_E164_DIGITS:
        return f"+{digits}"

    return None


def is_valid_e164_phone(phone: str | None) -> bool:
    """True si le numéro peut être normalisé en E.164 acceptable."""
    return normalize_e164_phone(phone) is not None


def mask_phone_for_log(phone: str | None) -> str:
    """Masque un numéro pour les logs (indicatif + 2 derniers chiffres)."""
    normalized = normalize_e164_phone(phone) or (phone or "").strip()
    digits = re.sub(r"\D", "", normalized)
    if len(digits) <= 2:
        return "inconnu" if not digits else "*" * len(digits)
    return f"+** *** *** {digits[-2:]}"
