# models/base.py
"""Helpers et utilitaires partagés par tous les models."""

from __future__ import annotations

import base64
import binascii
import os
from enum import Enum as PyEnum
from typing import TypeVar, cast

TEnum = TypeVar("TEnum", bound=PyEnum)

# --- Helpers de typage/normalisation ---


def _load_encryption_key() -> bytes:
    """Charge la clé d'encryption depuis les variables d'environnement."""
    b64 = (os.getenv("APP_ENCRYPTION_KEY_B64") or "").strip()
    if b64:
        padded = b64 + "=" * (-len(b64) % 4)
        try:
            key = base64.urlsafe_b64decode(padded.encode())
        except (binascii.Error, ValueError):
            msg = "APP_ENCRYPTION_KEY_B64 doit être en Base64 URL-safe."
            raise RuntimeError(msg) from None
        if len(key) not in (16, 24, 32):
            msg = "APP_ENCRYPTION_KEY_B64 invalide: longueur attendue 16/24/32 octets."
            raise RuntimeError(msg)
        return key

    legacy_hex = (
        os.getenv("ENCRYPTION_KEY_HEX") or os.getenv("ENCRYPTION_KEY") or ""
    ).strip()
    if legacy_hex:
        if legacy_hex.lower().startswith("0x"):
            legacy_hex = legacy_hex[2:]
        try:
            key = bytes.fromhex(legacy_hex)
        except ValueError:
            msg = "ENCRYPTION_KEY_HEX/ENCRYPTION_KEY invalide."
            raise RuntimeError(msg) from None
        if len(key) not in (16, 24, 32):
            msg = "Clé hex invalide: longueur attendue 16/24/32 octets."
            raise RuntimeError(msg)
        return key

    msg = "APP_ENCRYPTION_KEY_B64 manquante."
    raise RuntimeError(msg)


_encryption_key = _load_encryption_key()
_encryption_key_str = _encryption_key.hex()


# --- Helper functions pour conversion de types ---
def _as_bool(value):  # pyright: ignore[reportUnusedFunction]
    """Convertit une valeur en booléen."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return bool(value)


def _as_int(value):  # pyright: ignore[reportUnusedFunction]
    """Convertit une valeur en entier."""
    if value is None:
        return 0
    return int(value)


def _as_float(value):  # pyright: ignore[reportUnusedFunction]
    """Convertit une valeur en flottant."""
    if value is None:
        return 0.0
    return float(value)


def _as_str(value):  # pyright: ignore[reportUnusedFunction]
    """Convertit une valeur en string."""
    if value is None:
        return ""
    return str(value)


def _as_dt(value):  # pyright: ignore[reportUnusedFunction]
    """Convertit une valeur en datetime."""
    if value is None:
        return None
    from datetime import datetime

    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        from dateutil import parser

        return parser.parse(value)
    return value


def _iso(dt):  # pyright: ignore[reportUnusedFunction]
    """Convertit une datetime en string ISO."""
    if dt is None:
        return None
    return dt.isoformat()


def _coerce_enum(value, enum_class: type[TEnum]) -> TEnum:  # pyright: ignore[reportUnusedFunction]
    """Convertit une valeur en enum."""
    if isinstance(value, PyEnum):
        return cast(TEnum, value)
    if isinstance(value, str):
        try:
            return enum_class[value.upper()]
        except KeyError:
            return enum_class[value]
    return enum_class(value)
