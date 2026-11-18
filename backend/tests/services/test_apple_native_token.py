"""Le jeton Apple natif n'est pas un jeton Web."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from jwt.algorithms import ECAlgorithm

from services.apple_native_token import (
    NativeAppleTokenRejected,
    verify_native_apple_identity_token,
)
from services.external_auth import APPLE_ISSUER

_KEY = ec.generate_private_key(ec.SECP256R1())
_JWK = json.loads(ECAlgorithm.to_jwk(_KEY.public_key()))
_JWK.update({"kid": "apple-native-test", "alg": "ES256", "use": "sig"})
_BUNDLE = "ch.liri.operations"
_SERVICES = "ch.lirie.web.auth"


def _token(*, audience: str, nonce: str) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "iss": APPLE_ISSUER,
            "aud": audience,
            "sub": "apple-sub-stable",
            "nonce": nonce,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(minutes=5)).timestamp()),
        },
        _KEY,
        algorithm="ES256",
        headers={"kid": "apple-native-test"},
    )


def test_jeton_natif_accepte_lempreinte_du_nonce():
    raw = "nonce-brut-conserve-par-l-application"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    claims = verify_native_apple_identity_token(
        _token(audience=_BUNDLE, nonce=digest),
        bundle_id=_BUNDLE,
        raw_nonce=raw,
        jwks={"keys": [_JWK]},
    )
    assert claims["sub"] == "apple-sub-stable"
    assert claims["aud"] == _BUNDLE


def test_jeton_web_et_nonce_brut_sont_refuses():
    raw = "nonce-brut-conserve-par-l-application"
    with pytest.raises(NativeAppleTokenRejected) as web_token:
        verify_native_apple_identity_token(
            _token(audience=_SERVICES, nonce=raw),
            bundle_id=_BUNDLE,
            raw_nonce=raw,
            jwks={"keys": [_JWK]},
        )
    assert web_token.value.code == "jeton_invalide"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    with pytest.raises(NativeAppleTokenRejected) as raw_nonce:
        verify_native_apple_identity_token(
            _token(audience=_BUNDLE, nonce=raw),
            bundle_id=_BUNDLE,
            raw_nonce=raw,
            jwks={"keys": [_JWK]},
        )
    assert raw_nonce.value.code == "nonce_invalide"
    assert digest != raw
