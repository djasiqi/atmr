"""Jeton de handoff web à usage unique (quota appareils mobile → gestion web)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import quote

from flask import current_app, make_response
from flask_jwt_extended import create_access_token, create_refresh_token

from ext import db
from models import User
from security.refresh_token_service import store_refresh_token
from services.security.authentication import RefreshTokenService

logger = logging.getLogger(__name__)

HANDOFF_TOKEN_PREFIX = "web_handoff:"
HANDOFF_TOKEN_TTL_SECONDS = int(os.getenv("WEB_HANDOFF_TOKEN_TTL_SECONDS", "60"))


class WebHandoffError(Exception):
    def __init__(self, code: str, message: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.message = message or code


def _get_redis():
    try:
        from ext import redis_client

        return redis_client
    except Exception:
        return None


def _handoff_redis_key(token: str) -> str:
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"{HANDOFF_TOKEN_PREFIX}{digest}"


def build_device_management_redirect_path(user: User) -> str:
    """Chemin web canonique vers la section Sécurité selon le rôle."""
    role = user.role.value if user.role else "client"
    public_id = str(user.public_id)
    if role == "company":
        return f"/dashboard/company/{public_id}/settings#security"
    if role == "driver":
        return f"/dashboard/driver/{public_id}/settings#security"
    if role == "admin":
        return f"/dashboard/admin/{public_id}/settings#security"
    if role == "institution":
        return f"/dashboard/institution/{public_id}/settings#security"
    return f"/dashboard/{role}/{public_id}/settings#security"


def resolve_public_web_base_url() -> str:
    environment = str(current_app.config.get("ENVIRONMENT", "")).strip().lower()
    default_frontend_url = (
        "http://localhost:3000"
        if environment in {"development", "testing"}
        else "https://www.lirie.ch"
    )
    return (
        os.getenv("FRONTEND_URL")
        or os.getenv("PUBLIC_FRONTEND_URL")
        or os.getenv("PUBLIC_APP_URL")
        or default_frontend_url
    ).rstrip("/")


def validate_handoff_redirect_path(*, redirect_path: str, role: str) -> str:
    """Valide que le redirect est un chemin interne cohérent avec le rôle."""
    path = (redirect_path or "").strip()
    if not path.startswith("/") or "://" in path:
        raise WebHandoffError(
            "handoff_redirect_invalid",
            "Chemin de redirection invalide.",
        )
    normalized_role = (role or "").strip().lower()
    if normalized_role == "driver" and "/dashboard/company/" in path:
        raise WebHandoffError(
            "handoff_redirect_invalid",
            "Redirection entreprise interdite pour un compte chauffeur.",
        )
    if normalized_role == "company" and "/dashboard/driver/" in path:
        raise WebHandoffError(
            "handoff_redirect_invalid",
            "Redirection chauffeur interdite pour un compte entreprise.",
        )
    return path


def build_web_handoff_url(*, token: str) -> str:
    base = resolve_public_web_base_url()
    return f"{base}/auth/handoff?token={quote(token, safe='')}"


def issue_web_handoff_token(
    *,
    user_id: int,
    role: str | None,
    redirect_path: str,
    ttl_seconds: int | None = None,
) -> str | None:
    """Émet un jeton Redis à usage unique pour connexion web automatique."""
    r = _get_redis()
    if not r:
        logger.warning("issue_web_handoff_token: Redis indisponible")
        return None
    token = secrets.token_urlsafe(32)
    ttl = int(ttl_seconds or HANDOFF_TOKEN_TTL_SECONDS)
    payload = {
        "scope": "web_handoff",
        "user_id": int(user_id),
        "role": str(role or ""),
        "redirect_path": str(redirect_path),
        "issued_at": datetime.now(UTC).isoformat(),
    }
    try:
        r.setex(_handoff_redis_key(token), ttl, json.dumps(payload))
        return token
    except Exception as exc:
        logger.warning("issue_web_handoff_token failed: %s", exc)
        return None


def consume_web_handoff_token(*, token: str) -> dict[str, Any]:
    """Consomme le jeton (usage unique) et retourne le payload."""
    raw_token = (token or "").strip()
    if not raw_token:
        raise WebHandoffError("handoff_token_required", "Jeton de handoff manquant.")
    r = _get_redis()
    if not r:
        raise WebHandoffError(
            "handoff_unavailable",
            "Service de handoff temporairement indisponible.",
        )
    key = _handoff_redis_key(raw_token)
    try:
        raw = r.get(key)
        if not raw:
            raise WebHandoffError(
                "handoff_token_expired",
                "Le lien a expiré. Reconnectez-vous depuis l'application.",
            )
        r.delete(key)
        payload = json.loads(raw)
    except WebHandoffError:
        raise
    except Exception as exc:
        logger.warning("consume_web_handoff_token failed: %s", exc)
        raise WebHandoffError("handoff_unavailable") from exc
    if payload.get("scope") != "web_handoff":
        raise WebHandoffError("handoff_token_invalid")
    return payload


def _user_token_version(user: User) -> int:
    return int(getattr(user, "token_version", 0) or 0)


def _resolve_company_id(user: User) -> int | None:
    company_id = getattr(user, "company_id", None)
    if company_id is not None:
        return int(company_id)
    driver = getattr(user, "driver", None)
    if driver is not None and getattr(driver, "company_id", None) is not None:
        return int(driver.company_id)
    return None


def _get_password_hash_version(user: User) -> str:
    from routes.auth import _get_password_hash_version

    return _get_password_hash_version(user)


def _resolve_access_token_expires() -> timedelta:
    from routes.auth import _resolve_access_token_expires

    return _resolve_access_token_expires(is_mobile_request=False)


def _resolve_refresh_token_expires() -> timedelta:
    from routes.auth import _resolve_refresh_token_expires

    return _resolve_refresh_token_expires(is_mobile_request=False, remember_me=False)


def _resolve_max_active_refresh_tokens(user: User) -> int:
    from routes.auth import _resolve_max_active_refresh_tokens

    return _resolve_max_active_refresh_tokens(user)


def _access_expiry_metadata(access_expires_delta: timedelta) -> dict[str, object]:
    from routes.auth import _access_expiry_metadata

    return _access_expiry_metadata(access_expires_delta)


def _clear_web_auth_cookies(response) -> None:
    from routes.auth import _clear_web_auth_cookies

    _clear_web_auth_cookies(response)


def create_web_handoff_session_response(
    user: User,
    *,
    redirect_path: str,
    trace_id: str | None = None,
):
    """Crée une session web (cookies HttpOnly) après consommation du handoff."""
    from routes.auth import _must_complete_onboarding, _onboarding_reasons

    claims = {
        "role": user.role.value if user.role else None,
        "company_id": _resolve_company_id(user),
        "driver_id": getattr(user, "driver_id", None),
        "institution_id": getattr(user, "institution_id", None),
        "institution_role": getattr(user, "institution_role", None),
        "aud": "atmr-api",
        "token_version": _user_token_version(user),
    }
    access_expires_delta = _resolve_access_token_expires()
    access_token = create_access_token(
        identity=str(user.public_id),
        additional_claims=claims,
        expires_delta=access_expires_delta,
        fresh=True,
    )
    refresh_expires_delta = _resolve_refresh_token_expires()
    pwd_hash_version = _get_password_hash_version(user)
    refresh_claims: dict[str, object] = {
        "aud": "atmr-api",
        "pwd_hash": pwd_hash_version,
        "token_version": _user_token_version(user),
        "remember_me": False,
    }
    refresh_token = create_refresh_token(
        identity=str(user.public_id),
        additional_claims=refresh_claims,
        expires_delta=refresh_expires_delta,
    )

    refresh_expires_at = datetime.now(UTC) + refresh_expires_delta
    store_refresh_token(
        token=refresh_token,
        user_id=user.id,
        expires_at=refresh_expires_at,
        device_id=None,
        device_name="Web handoff",
        commit=False,
    )
    db.session.commit()

    token_service = RefreshTokenService()
    with suppress(Exception):
        token_service.store_token(
            user.id,
            refresh_token,
            ttl_seconds=int(refresh_expires_delta.total_seconds()),
        )
    with suppress(Exception):
        token_service.limit_active_tokens(
            user.id, _resolve_max_active_refresh_tokens(user)
        )

    response_data: dict[str, object] = {
        "message": "Handoff réussi",
        "user": {
            "id": user.id,
            "public_id": user.public_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value if user.role else None,
            "force_password_change": user.force_password_change,
            "must_complete_onboarding": _must_complete_onboarding(user),
            "onboarding_reasons": _onboarding_reasons(user),
        },
        "redirect_to": redirect_path,
        "token": access_token,
        "refresh_token": refresh_token,
        "target_env": current_app.config.get("ENVIRONMENT", "production"),
        "trace_id": trace_id,
    }
    response_data.update(_access_expiry_metadata(access_expires_delta))

    response = make_response(response_data, 200)
    _clear_web_auth_cookies(response)
    response.set_cookie(
        current_app.config["COOKIE_ACCESS_TOKEN_NAME"],
        access_token,
        httponly=current_app.config["COOKIE_HTTP_ONLY"],
        secure=current_app.config["COOKIE_SECURE"],
        samesite=current_app.config["COOKIE_SAME_SITE"],
        max_age=int(current_app.config["JWT_ACCESS_TOKEN_EXPIRES"].total_seconds()),
        path=current_app.config["COOKIE_PATH"],
        domain=current_app.config["COOKIE_DOMAIN"],
    )
    response.set_cookie(
        current_app.config["COOKIE_REFRESH_TOKEN_NAME"],
        refresh_token,
        httponly=current_app.config["COOKIE_HTTP_ONLY"],
        secure=current_app.config["COOKIE_SECURE"],
        samesite=current_app.config["COOKIE_SAME_SITE"],
        max_age=None,
        path=current_app.config["COOKIE_PATH"],
        domain=current_app.config["COOKIE_DOMAIN"],
    )
    return response
