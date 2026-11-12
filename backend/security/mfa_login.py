"""Interruption login MFA : challenge ou enrollment, jamais de JWT métier."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from flask_jwt_extended import create_access_token, decode_token

from security.privileged_access import requires_mfa_enrollment
from security.totp_service import (
    store_2fa_challenge_jti,
    store_mfa_enroll_jti,
)

PURPOSE_2FA_CHALLENGE = "2fa_challenge"
PURPOSE_MFA_ENROLL = "mfa_enroll"
RESTRICTED_JWT_PURPOSES = frozenset({PURPOSE_2FA_CHALLENGE, PURPOSE_MFA_ENROLL})
MFA_TEMP_TOKEN_TTL = timedelta(minutes=5)
MFA_ENROLL_TOKEN_TTL = timedelta(minutes=15)


def is_restricted_mfa_jwt(payload: dict[str, Any] | None) -> bool:
    """True si le JWT est un temp_token MFA, pas un access/refresh métier."""
    if not payload:
        return False
    return payload.get("purpose") in RESTRICTED_JWT_PURPOSES


def evaluate_login_mfa(user: Any) -> str | None:
    """Retourne le purpose MFA si le login doit s'arrêter, sinon None."""
    if user is None:
        return None
    if bool(getattr(user, "totp_enabled", False)):
        return PURPOSE_2FA_CHALLENGE
    if requires_mfa_enrollment(user):
        return PURPOSE_MFA_ENROLL
    return None


def _user_token_version(user: Any) -> int:
    return int(getattr(user, "token_version", 0) or 0)


def issue_mfa_temp_token(
    user: Any,
    purpose: str,
    *,
    remember_me: bool = False,
) -> str:
    """Émet un JWT à purpose restreint. Inutilisable comme access métier."""
    if purpose not in RESTRICTED_JWT_PURPOSES:
        raise ValueError(f"purpose MFA inconnu: {purpose}")
    ttl = MFA_ENROLL_TOKEN_TTL if purpose == PURPOSE_MFA_ENROLL else MFA_TEMP_TOKEN_TTL
    token = create_access_token(
        identity=str(user.public_id),
        additional_claims={
            "purpose": purpose,
            "aud": "atmr-api",
            "remember_me": bool(remember_me),
            "token_version": _user_token_version(user),
        },
        expires_delta=ttl,
        fresh=False,
    )
    decoded = decode_token(token)
    jti = decoded.get("jti")
    if jti:
        if purpose == PURPOSE_2FA_CHALLENGE:
            store_2fa_challenge_jti(str(jti))
        else:
            store_mfa_enroll_jti(str(jti))
    return token


def build_mfa_interrupt_response(
    user: Any,
    purpose: str,
    *,
    remember_me: bool = False,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    """Réponse 202 : mot de passe OK, aucun JWT métier."""
    from middleware.trace_id import get_trace_id

    role = getattr(user, "role", None)
    payload: dict[str, Any] = {
        "mfa_required": True,
        "mfa_purpose": purpose,
        "temp_token": issue_mfa_temp_token(user, purpose, remember_me=remember_me),
        "error_code": (
            "mfa_challenge_required"
            if purpose == PURPOSE_2FA_CHALLENGE
            else "mfa_enroll_required"
        ),
        "message": (
            "Code de validation en deux étapes requis."
            if purpose == PURPOSE_2FA_CHALLENGE
            else "Activation de la validation en deux étapes obligatoire."
        ),
        "user": {
            "public_id": getattr(user, "public_id", None),
            "role": role.value if hasattr(role, "value") else role,
        },
        "trace_id": get_trace_id(),
    }
    if extra:
        payload.update(extra)
    return payload, 202


def maybe_interrupt_login_for_mfa(
    user: Any,
    *,
    remember_me: bool = False,
) -> tuple[dict[str, Any], int] | None:
    purpose = evaluate_login_mfa(user)
    if purpose is None:
        return None
    return build_mfa_interrupt_response(user, purpose, remember_me=remember_me)
