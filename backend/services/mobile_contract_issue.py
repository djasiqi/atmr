"""Émet le contrat mobile déjà utilisé par le mot de passe.

Aucun cookie Web n'est posé. Le corps reprend les champs exigés par
``mobile-device-session-v1``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from flask import request
from flask_jwt_extended import create_access_token, create_refresh_token

from models import User
from shared.response_helpers import json_response


class MobileContractError(Exception):
    """Refus du contrat d'appareil, sans jeton dans la réponse."""

    def __init__(self, body: dict, status: int) -> None:
        super().__init__(body.get("error_code") or "contrat_mobile")
        self.body = body
        self.status = status


def issue_mobile_device_contract(user: User):
    """Crée la session d'appareil puis les jetons LIRIE habituels."""
    from routes.auth import (
        _get_password_hash_version,
        _must_complete_onboarding,
        _onboarding_reasons,
        _resolve_access_token_expires,
        _resolve_company_id,
        _resolve_device_session_metadata,
        _resolve_refresh_token_expires,
        _user_token_version,
    )
    from security.mobile_device_session_service import (
        DeviceSessionLimitReached,
        auth_capabilities,
        create_or_reuse_session,
    )
    from security.refresh_redis_rotation import (
        commit_db_after_redis,
        publish_refresh_redis,
    )
    from security.refresh_token_service import (
        RefreshStoreUnavailableError,
        store_refresh_token,
    )

    device_id = (
        request.headers.get("X-Device-ID") or request.headers.get("X-Installation-ID") or ""
    ).strip()
    contract = (request.headers.get("X-Auth-Contract-Version") or "").strip()
    if contract != "mobile-device-session-v1" or not device_id:
        raise MobileContractError(
            {
                "error": "device_identity_required",
                "error_code": "device_identity_required",
                "message": "Identité sécurisée de l'appareil manquante.",
            },
            400,
        )
    driver_id = getattr(user, "driver_id", None)
    if driver_id is None:
        driver_id = getattr(getattr(user, "driver", None), "id", None)
    try:
        mobile_session, recovery, revocation, _reaped = create_or_reuse_session(
            user_id=int(user.id),
            device_installation_id=device_id,
            driver_id=driver_id,
            role=user.role.value if user.role else None,
            meta=_resolve_device_session_metadata(),
        )
    except DeviceSessionLimitReached:
        raise MobileContractError(
            {
                "error": "device_session_limit_reached",
                "error_code": "device_session_limit_reached",
                "message": (
                    "Nombre maximal d'appareils atteint. "
                    "Déconnectez un ancien appareil pour continuer."
                ),
            },
            409,
        ) from None
    if mobile_session is None or not recovery or not revocation:
        raise MobileContractError(
            {
                "error": "mobile_session_contract_incomplete",
                "error_code": "mobile_session_contract_incomplete",
            },
            503,
        )
    refresh_expires = _resolve_refresh_token_expires(
        is_mobile_request=True, remember_me=False
    )
    access_expires = _resolve_access_token_expires(True)
    epoch = int(getattr(mobile_session, "session_epoch", 1) or 1)
    refresh_generation = int(getattr(mobile_session, "refresh_generation", 1) or 1)
    claims = {
        "role": user.role.value,
        "company_id": _resolve_company_id(user),
        "driver_id": getattr(user, "driver_id", None),
        "institution_id": getattr(user, "institution_id", None),
        "institution_role": getattr(user, "institution_role", None),
        "aud": "atmr-api",
        "token_version": _user_token_version(user),
        "session_id": str(mobile_session.session_id),
        "session_epoch": epoch,
        "session_generation": epoch,
    }
    access_token = create_access_token(
        identity=str(user.public_id),
        additional_claims=claims,
        expires_delta=access_expires,
        fresh=True,
    )
    refresh_token = create_refresh_token(
        identity=str(user.public_id),
        additional_claims={
            "aud": "atmr-api",
            "pwd_hash": _get_password_hash_version(user),
            "token_version": _user_token_version(user),
            "session_id": str(mobile_session.session_id),
            "session_epoch": epoch,
            "refresh_generation": refresh_generation,
            "session_generation": epoch,
        },
        expires_delta=refresh_expires,
    )
    try:
        stored = store_refresh_token(
            token=refresh_token,
            user_id=int(user.id),
            expires_at=datetime.now(UTC) + refresh_expires,
            device_id=device_id,
            device_name=request.headers.get("X-Device-Name"),
            commit=False,
            sync_redis=False,
        )
        if hasattr(stored, "session_id"):
            stored.session_id = str(mobile_session.session_id)
            stored.session_generation = epoch
        redis_handle = publish_refresh_redis(
            int(user.id),
            refresh_token,
            ttl_seconds=int(refresh_expires.total_seconds()),
        )
        commit_db_after_redis(redis_handle)
    except RefreshStoreUnavailableError as error:
        from ext import db

        db.session.rollback()
        raise MobileContractError(
            {
                "error": "service_unavailable",
                "error_code": "store_unavailable",
                "message": "Stockage session indisponible. Réessayez.",
            },
            503,
        ) from error
    body = {
        "message": "Connexion réussie",
        "token": access_token,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "session_id": str(mobile_session.session_id),
        "session_epoch": epoch,
        "credential_generation": int(
            getattr(mobile_session, "credential_generation", mobile_session.generation)
            or 1
        ),
        "refresh_generation": refresh_generation,
        "session_generation": epoch,
        "recovery_credential": recovery,
        "revocation_secret": revocation,
        "user": {
            "id": user.id,
            "public_id": user.public_id,
            "username": user.username,
            "email": user.email,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "role": user.role.value,
            "force_password_change": bool(user.force_password_change),
            "must_complete_onboarding": _must_complete_onboarding(user),
            "onboarding_reasons": _onboarding_reasons(user),
        },
    }
    body.update(auth_capabilities())
    return json_response(body, 200)
