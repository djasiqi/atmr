# models/base.py
"""
Helpers et utilitaires partagés par tous les models.
"""
from __future__ import annotations

import base64
import binascii
import os
from datetime import datetime
from enum import Enum as PyEnum
from typing import Any, Type, TypeVar

TEnum = TypeVar("TEnum", bound=PyEnum)

# --- Helpers de typage/normalisation ---
def _as_dt(v: Any) -> datetime | None:
    return v if isinstance(v, datetime) else None

def _as_str(v: Any) -> str | None:
    return v if isinstance(v, str) else None

def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default

def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default

def _as_bool(v: Any) -> bool:
    return bool(v) if isinstance(v, (bool, int)) else False

def _iso(v: Any) -> str | None:
    dt = _as_dt(v)
    return dt.isoformat() if dt else None

def _coerce_enum(v: Any, enum_cls: Type[TEnum]) -> TEnum | None:
    """Transforme str → Enum (par value OU par name), sinon None."""
    if isinstance(v, enum_cls):
        return v
    if isinstance(v, str):
        try:
            return enum_cls(v)
        except Exception:
            try:
                return enum_cls[v]
            except Exception:
                return None
    return None

def _load_encryption_key() -> bytes:
    """Charge la clé d'encryption depuis les variables d'environnement."""
    b64 = (os.getenv("APP_ENCRYPTION_KEY_B64") or "").strip()
    if b64:
        padded = b64 + "=" * (-len(b64) % 4)
        try:
            key = base64.urlsafe_b64decode(padded.encode())
        except (binascii.Error, ValueError):
            raise RuntimeError("APP_ENCRYPTION_KEY_B64 doit être en Base64 URL-safe.")
        if len(key) not in (16, 24, 32):
            raise RuntimeError("APP_ENCRYPTION_KEY_B64 invalide: longueur attendue 16/24/32 octets.")
        return key

    legacy_hex = (os.getenv("ENCRYPTION_KEY_HEX") or os.getenv("ENCRYPTION_KEY") or "").strip()
    if legacy_hex:
        if legacy_hex.lower().startswith("0x"):
            legacy_hex = legacy_hex[2:]
        try:
            key = bytes.fromhex(legacy_hex)
        except ValueError:
            raise RuntimeError("ENCRYPTION_KEY_HEX/ENCRYPTION_KEY invalide.")
        if len(key) not in (16, 24, 32):
            raise RuntimeError("Clé hex invalide: longueur attendue 16/24/32 octets.")
        return key

    raise RuntimeError("APP_ENCRYPTION_KEY_B64 manquante.")

_encryption_key = _load_encryption_key()
_encryption_key_str = _encryption_key.hex()

