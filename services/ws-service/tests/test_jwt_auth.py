"""Sécurité JWT Socket.IO — audiences LIRIE + access vs refresh."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest

from auth_claims import normalize_auth_payload
from jwt_auth import ALLOWED_JWT_AUDIENCES, decode_socket_access_token

SECRET = "unit-test-secret-not-for-production"
ALG = "HS256"


def _encode(
    claims: dict,
    *,
    secret: str = SECRET,
    algorithm: str = ALG,
    headers: dict | None = None,
) -> str:
    return jwt.encode(claims, secret, algorithm=algorithm, headers=headers)


def _base_access(**extra) -> dict:
    now = datetime.now(UTC)
    payload = {
        "sub": "550e8400-e29b-41d4-a716-446655440000",
        "user_id": 5,
        "role": "COMPANY",
        "company_id": 1,
        "type": "access",
        "aud": "atmr-api",
        "iat": now,
        "exp": now + timedelta(minutes=30),
    }
    payload.update(extra)
    return payload


def test_allowed_audiences_match_backend_contract() -> None:
    assert set(ALLOWED_JWT_AUDIENCES) == {"atmr-api", "atmr-mobile-enterprise"}


def test_accept_aud_atmr_api() -> None:
    token = _encode(_base_access(aud="atmr-api"))
    payload = decode_socket_access_token(token, secret=SECRET, algorithm=ALG)
    assert payload is not None
    assert payload["aud"] == "atmr-api"
    assert normalize_auth_payload(payload) is not None


def test_accept_aud_atmr_mobile_enterprise() -> None:
    token = _encode(_base_access(aud="atmr-mobile-enterprise", role="company"))
    payload = decode_socket_access_token(token, secret=SECRET, algorithm=ALG)
    assert payload is not None
    assert payload["aud"] == "atmr-mobile-enterprise"
    claims = normalize_auth_payload(payload)
    assert claims is not None
    assert claims["company_id"] == 1
    assert claims["role"] == "company"


def test_reject_missing_aud() -> None:
    claims = _base_access()
    del claims["aud"]
    token = _encode(claims)
    assert decode_socket_access_token(token, secret=SECRET, algorithm=ALG) is None


def test_reject_invalid_aud() -> None:
    token = _encode(_base_access(aud="evil-service"))
    assert decode_socket_access_token(token, secret=SECRET, algorithm=ALG) is None


def test_reject_expired() -> None:
    now = datetime.now(UTC)
    token = _encode(
        _base_access(
            iat=now - timedelta(hours=2),
            exp=now - timedelta(hours=1),
        )
    )
    assert decode_socket_access_token(token, secret=SECRET, algorithm=ALG) is None


def test_reject_bad_signature() -> None:
    token = _encode(_base_access())
    assert (
        decode_socket_access_token(
            token,
            secret="wrong-secret-at-least-32-bytes-long!!",
            algorithm=ALG,
        )
        is None
    )


def test_reject_unauthorized_algorithm() -> None:
    """alg=none (ou autre hors HS256) doit être refusé."""
    claims = _base_access()
    # PyJWT refuse d'encoder avec alg=none sans clés ; on force le header.
    token = jwt.encode(claims, key="", algorithm="none")
    assert decode_socket_access_token(token, secret=SECRET, algorithm=ALG) is None


def test_reject_refresh_token_even_with_valid_aud_and_role() -> None:
    """Contrat Flask-JWT-Extended : type=refresh — ne pas dépendre de l'absence de role.

    Le refresh mobile entreprise peut porter aud + session_id seulement, ou
    (switch_to_enterprise) aud + role. Dans les deux cas, type=refresh ⇒ reject.
    """
    now = datetime.now(UTC)
    mobile_refresh = {
        "sub": "550e8400-e29b-41d4-a716-446655440000",
        "aud": "atmr-mobile-enterprise",
        "session_id": "sess-1",
        "type": "refresh",
        "iat": now,
        "exp": now + timedelta(days=30),
    }
    token = _encode(mobile_refresh)
    assert decode_socket_access_token(token, secret=SECRET, algorithm=ALG) is None

    refresh_with_role = {
        **mobile_refresh,
        "role": "company",
        "company_id": 1,
    }
    token2 = _encode(refresh_with_role)
    assert decode_socket_access_token(token2, secret=SECRET, algorithm=ALG) is None
    # Effet secondaire documenté : normalize pourrait accepter le payload décodé
    # si on contournait type — d'où le contrôle explicite type=access.
    assert normalize_auth_payload(refresh_with_role) is not None


def test_reject_missing_type_claim() -> None:
    claims = _base_access()
    del claims["type"]
    token = _encode(claims)
    assert decode_socket_access_token(token, secret=SECRET, algorithm=ALG) is None
