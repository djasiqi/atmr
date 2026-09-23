"""Validation JWT Socket.IO alignée sur le contrat backend LIRIE.

Audiences acceptées (identique à ``validate_jwt_audience`` backend) :

- ``atmr-api``
- ``atmr-mobile-enterprise``

Seul un access token Flask-JWT-Extended (``type=access``) est accepté.
"""

from __future__ import annotations

from typing import Any

import jwt

# Aligné sur backend/ext.py::validate_jwt_audience
ALLOWED_JWT_AUDIENCES: tuple[str, ...] = ("atmr-api", "atmr-mobile-enterprise")
ACCESS_TOKEN_TYPE = "access"


def decode_socket_access_token(
    token: str,
    *,
    secret: str,
    algorithm: str,
) -> dict[str, Any] | None:
    """Décode et valide un access token pour la connexion Socket.IO.

    Utilise la validation native PyJWT (signature, ``exp``, ``aud``).
    Rejette ensuite tout token dont ``type`` n'est pas ``access``
    (refresh Flask-JWT-Extended : ``type=refresh``).

    Ne journalise jamais le token.
    """
    if not token or not secret or not algorithm:
        return None

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            audience=list(ALLOWED_JWT_AUDIENCES),
        )
    except jwt.PyJWTError:
        return None

    if not isinstance(payload, dict):
        return None

    # Contrat Flask-JWT-Extended : access → type=access, refresh → type=refresh
    if payload.get("type") != ACCESS_TOKEN_TYPE:
        return None

    return payload
