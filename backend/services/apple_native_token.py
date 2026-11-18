"""Vérifie un jeton d'identité Sign in with Apple natif.

Ce module ne crée pas d'utilisateur, n'émet pas de session et ne modifie
pas le parcours Web. L'audience attendue est l'identifiant de bundle,
jamais le Services ID du site.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import jwt
from jwt.algorithms import ECAlgorithm

from services.external_auth import APPLE_ISSUER


class NativeAppleTokenRejected(Exception):
    """Jeton natif refusé, sans session."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def verify_native_apple_identity_token(
    token: str,
    *,
    bundle_id: str,
    raw_nonce: str,
    jwks: dict[str, Any],
) -> dict[str, Any]:
    """Contrôle signature, issuer, audience du bundle, expiration et nonce.

    Le client natif transmet à Apple l'empreinte SHA-256 hexadécimale du
    nonce, et conserve la valeur brute pour le serveur. Un jeton Web, dont
    l'audience est le Services ID et dont le nonce est brut, ne passe pas.
    """
    audience = str(bundle_id or "").strip()
    nonce = str(raw_nonce or "").strip()
    if not audience or not nonce or not str(token or "").strip():
        raise NativeAppleTokenRejected("jeton_invalide")
    try:
        header = jwt.get_unverified_header(token)
    except jwt.PyJWTError as error:
        raise NativeAppleTokenRejected("jeton_invalide") from error
    if str(header.get("alg") or "") != "ES256":
        raise NativeAppleTokenRejected("jeton_invalide")
    key = _ec_key(jwks, str(header.get("kid") or ""))
    try:
        claims = jwt.decode(
            token,
            key,
            algorithms=["ES256"],
            audience=audience,
            issuer=APPLE_ISSUER,
            leeway=60,
            options={"require": ["exp", "iat", "sub", "iss", "aud", "nonce"]},
        )
    except jwt.PyJWTError as error:
        raise NativeAppleTokenRejected("jeton_invalide") from error
    expected = hashlib.sha256(nonce.encode("utf-8")).hexdigest()
    if str(claims.get("nonce") or "") != expected:
        raise NativeAppleTokenRejected("nonce_invalide")
    if str(claims.get("aud") or "") != audience:
        raise NativeAppleTokenRejected("jeton_invalide")
    return claims


def _ec_key(jwks: dict[str, Any], kid: str) -> Any:
    keys = jwks.get("keys") if isinstance(jwks, dict) else None
    if not isinstance(keys, list):
        raise NativeAppleTokenRejected("jeton_invalide")
    chosen = next((item for item in keys if str(item.get("kid") or "") == kid), None)
    if not isinstance(chosen, dict) or chosen.get("kty") != "EC":
        raise NativeAppleTokenRejected("jeton_invalide")
    return ECAlgorithm.from_jwk(json.dumps(chosen))
