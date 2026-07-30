"""Décorateur / helpers de validation de session mobile durable."""

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

from flask import g, jsonify
from flask_jwt_extended import get_jwt, verify_jwt_in_request

from security.mobile_device_session_service import validate_mobile_session

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def extract_mobile_session_claims(claims: dict[str, Any] | None = None) -> tuple[str | None, int | None]:
    data = claims if claims is not None else {}
    try:
        data = data or get_jwt() or {}
    except Exception:
        data = data or {}
    session_id = data.get("session_id")
    if session_id is not None:
        session_id = str(session_id)
    raw_gen = data.get("session_generation")
    try:
        session_generation = int(raw_gen) if raw_gen is not None else None
    except (TypeError, ValueError):
        session_generation = None
    return session_id, session_generation


def mobile_session_required(fn: F) -> F:
    """Valide session_id + generation sur les endpoints protégés mobile.

    Compat : si le JWT n'a pas de session_id (clients legacy), laisse passer.
    Si validation serveur indisponible → 503 retryable (pas de purge client).
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        try:
            verify_jwt_in_request(optional=False)
        except Exception:
            return jsonify({"error": "unauthorized", "error_code": "unauthorized"}), 401

        claims = get_jwt() or {}
        session_id, session_generation = extract_mobile_session_claims(claims)
        user_id = None
        try:
            from models import User

            identity = claims.get("sub") or claims.get("identity")
            # Flask-JWT-Extended stocke souvent l'identity séparément
            from flask_jwt_extended import get_jwt_identity

            identity = get_jwt_identity() or identity
            if identity:
                user = User.query.filter_by(public_id=str(identity)).first()
                if user:
                    user_id = user.id
                    # token_version
                    jwt_tv = claims.get("token_version")
                    if jwt_tv is not None and int(getattr(user, "token_version", 0) or 0) != int(jwt_tv):
                        return (
                            jsonify(
                                {
                                    "error": "token_version_mismatch",
                                    "error_code": "token_version_mismatch",
                                }
                            ),
                            401,
                        )
        except Exception as exc:
            logger.debug("mobile_session user resolve: %s", exc)

        error_code, retryable = validate_mobile_session(
            session_id=session_id,
            session_generation=session_generation,
            user_id=user_id,
        )
        if error_code == "session_validation_unavailable":
            return (
                jsonify(
                    {
                        "error": error_code,
                        "error_code": error_code,
                        "retryable": True,
                    }
                ),
                503,
            )
        if error_code:
            return (
                jsonify(
                    {
                        "error": error_code,
                        "error_code": error_code,
                        "retryable": False,
                    }
                ),
                401,
            )

        g.mobile_session_id = session_id
        g.mobile_session_generation = session_generation
        return fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def check_mobile_session_from_claims(
    claims: dict[str, Any], *, user_id: int | None = None
) -> tuple[str | None, bool]:
    """Pour handshake WebSocket / chemins hors décorateur Flask."""
    session_id, session_generation = extract_mobile_session_claims(claims)
    return validate_mobile_session(
        session_id=session_id,
        session_generation=session_generation,
        user_id=user_id,
    )
