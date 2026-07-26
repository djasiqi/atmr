"""Jeton d'activation HMAC versionné (Lot 1)."""

from __future__ import annotations

import hashlib
import hmac
import os


class ActivationTokenKeyError(RuntimeError):
    """Clé ACTIVATION_TOKEN_KEY_Vn manquante ou invalide."""


def _key_env_name(version: int) -> str:
    return f"ACTIVATION_TOKEN_KEY_V{int(version)}"


def get_activation_token_key(version: int = 1) -> bytes:
    """Retourne la clé pour une version ; obligatoire en production."""
    name = _key_env_name(version)
    raw = (os.getenv(name) or "").strip()
    env = (os.getenv("ENVIRONMENT") or "").strip().lower()
    if not raw:
        if env == "production":
            raise ActivationTokenKeyError(
                f"{name} obligatoire en production (distincte JWT/CSRF)"
            )
        # Dev/test : dérivation non secrète pour ne pas bloquer les suites
        raw = f"dev-only-{name}-not-for-production"
    return raw.encode("utf-8")


def require_activation_token_key_in_production() -> None:
    env = (os.getenv("ENVIRONMENT") or "").strip().lower()
    if env != "production":
        return
    get_activation_token_key(1)


def derive_activation_token(email_delivery_id: str, *, key_version: int = 1) -> str:
    """token = hex(HMAC-SHA256(ACTIVATION_TOKEN_KEY_Vn, email_delivery_id))."""
    key = get_activation_token_key(key_version)
    return hmac.new(
        key,
        str(email_delivery_id).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def hash_activation_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_activation_token(
    token: str,
    email_delivery_id: str,
    *,
    key_version: int = 1,
) -> bool:
    expected = derive_activation_token(email_delivery_id, key_version=key_version)
    return hmac.compare_digest(expected, token)
