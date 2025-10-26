# models/base.py
"""Helpers et utilitaires partagés par tous les models."""
from __future__ import annotations

import base64
import binascii
import os
from enum import Enum as PyEnum
from typing import TypeVar

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

    legacy_hex = (os.getenv("ENCRYPTION_KEY_HEX")
                  or os.getenv("ENCRYPTION_KEY") or "").strip()
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
