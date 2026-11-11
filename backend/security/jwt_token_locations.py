"""Lot 1-E / AUTH-PRO-4C : contrat JWT web (cookies) vs mobile (Bearer).

Séparation explicite par préfixe d'API et présence d'``Authorization: Bearer``,
sans s'appuyer sur le User-Agent comme mécanisme de sécurité pour le portail
chauffeur web.
"""

from __future__ import annotations


def resolve_jwt_token_locations(
    path: str,
    *,
    authorization: str | None = None,
    x_requested_with: str | None = None,
    user_agent: str | None = None,
) -> list[str]:
    """Retourne ``JWT_TOKEN_LOCATION`` pour la requête courante.

    Contrats :
    - ``/api/v1/company_mobile/`` : Bearer headers uniquement.
    - ``/api/v1/driver/`` + ``Authorization: Bearer`` : headers uniquement
      (app mobile / clients Bearer ; cookies web ignorés).
    - ``/api/v1/driver/`` sans Bearer : cookies puis headers (portail web).
    - Clients Expo explicites (``X-Requested-With: Expo``) : headers uniquement.
    - Autres chemins : cookies puis headers (web).

    Le User-Agent mobile historique reste un filet pour les clients Expo/okhttp
    hors préfixe driver ; il n'est pas utilisé pour décider le contrat driver web.
    """
    normalized = path or ""
    auth = (authorization or "").strip()
    has_bearer = auth.lower().startswith("bearer ")

    if normalized.startswith("/api/v1/company_mobile/"):
        return ["headers"]

    if normalized.startswith("/api/v1/driver/"):
        if has_bearer:
            return ["headers"]
        return ["cookies", "headers"]

    if (x_requested_with or "").strip() == "Expo":
        return ["headers"]

    ua = (user_agent or "").lower()
    if any(marker in ua for marker in ("expo", "okhttp", "darwin", "cfnetwork", "dalvik")):
        return ["headers"]

    return ["cookies", "headers"]
