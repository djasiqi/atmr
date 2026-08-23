"""Détection du type de client HTTP (web vs mobile) — factorisation partagée."""

from __future__ import annotations

from flask import request

_MOBILE_UA_MARKERS = (
    "okhttp",
    "cfnetwork",
    "darwin",
    "iphone",
    "ipad",
    "android",
    "mobile",
    "lirioprations",
    "lirioperations",
)


def is_mobile_request_client() -> bool:
    """Détecte une requête mobile (app native ou UA mobile)."""
    if request.headers.get("X-Requested-With") == "Expo":
        return True
    client_platform = (request.headers.get("X-Client-Platform") or "").strip().lower()
    if client_platform in {"ios", "android"}:
        return True
    user_agent = (request.headers.get("User-Agent") or "").lower()
    return any(marker in user_agent for marker in _MOBILE_UA_MARKERS)
