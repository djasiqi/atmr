"""Validation des URL de retour MyCheckout (prévention open redirect)."""

from __future__ import annotations

import os
from urllib.parse import urlsplit

from shared.return_url_policy import (
    http_return_matches_allowed_base,
    strip_return_url_base,
)


def allowed_return_url_prefixes() -> list[str]:
    """Bases autorisées pour un `return_url` fourni par le client (JSON body)."""
    prefixes: list[str] = []
    for key in ("CLIENT_WEB_BASE_URL", "PUBLIC_BASE_URL"):
        v = (os.getenv(key) or "").strip()
        if v:
            prefixes.append(strip_return_url_base(v))
    raw = (os.getenv("WORLDLINE_ALLOWED_RETURN_URL_PREFIXES") or "").strip()
    if raw:
        for part in raw.split(","):
            p = strip_return_url_base(part)
            if p:
                prefixes.append(p)
    seen: set[str] = set()
    out: list[str] = []
    for p in prefixes:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def validate_return_url_override(url: str) -> str:
    """Vérifie qu'une URL de retour explicite est autorisée.

    Raises:
        ValueError: schéma invalide, https requis, ou origine non autorisée.
    """
    u = (url or "").strip()
    if not u:
        raise ValueError("return_url ne peut pas être vide")

    parsed = urlsplit(u)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("return_url doit être une URL absolue (http/https)")
    if parsed.username or parsed.password:
        raise ValueError("return_url non autorisée par la configuration serveur")

    require_https = (
        os.getenv("WORLDLINE_RETURN_URL_REQUIRE_HTTPS") or ""
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if require_https and parsed.scheme != "https":
        raise ValueError("return_url doit utiliser https")

    prefixes = allowed_return_url_prefixes()
    if not prefixes:
        raise ValueError(
            "return_url personnalisée interdite: définir CLIENT_WEB_BASE_URL, "
            "PUBLIC_BASE_URL ou WORLDLINE_ALLOWED_RETURN_URL_PREFIXES"
        )

    if not any(http_return_matches_allowed_base(u, base) for base in prefixes):
        raise ValueError("return_url non autorisée par la configuration serveur")

    return u


def default_worldline_return_url(booking_id: int) -> str:
    """URL de retour construite côté serveur (hors champ return_url du client)."""
    base = (os.getenv("CLIENT_WEB_BASE_URL") or "").strip().rstrip("/")
    if not base:
        base = (
            (os.getenv("PUBLIC_BASE_URL") or "http://localhost:3000")
            .strip()
            .rstrip("/")
        )
    return f"{base}/client/payment/worldline/return?bookingId={booking_id}"
