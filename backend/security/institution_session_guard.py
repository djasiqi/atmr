"""Garde session institution — validation sid sur chaque JWT institution."""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from flask import g

from security.web_session_service import (
    SESSION_UPGRADE_REQUIRED,
    claims_require_institution_session_guard,
    extract_sid_from_claims,
    validate_web_session_for_request,
)

logger = logging.getLogger(__name__)


def _mark_jwt_revocation_reason(reason: str) -> None:
    with contextlib.suppress(RuntimeError):
        g.jwt_revocation_reason = reason


def resolve_user_id_from_jwt_payload(jwt_payload: dict[str, Any]) -> int | None:
    try:
        from models import User

        identity = jwt_payload.get("sub")
        if not identity:
            return None
        user = User.query.filter_by(public_id=str(identity)).first()
        return user.id if user else None
    except Exception as exc:
        logger.debug("institution_session_guard user resolve: %s", exc)
        return None


def is_institution_jwt_revoked_by_session(jwt_payload: dict[str, Any]) -> bool:
    """Retourne True si le JWT doit être rejeté (session révoquée ou idle).

    Appelé depuis le pipeline JWT (token_in_blocklist_loader).
    Ne met jamais à jour last_interactive_activity_at.
    """
    if not claims_require_institution_session_guard(jwt_payload):
        return False

    sid = extract_sid_from_claims(jwt_payload)
    if not sid:
        logger.info("institution_session_guard reject missing sid")
        _mark_jwt_revocation_reason(SESSION_UPGRADE_REQUIRED)
        return True

    user_id = resolve_user_id_from_jwt_payload(jwt_payload)
    error_code = validate_web_session_for_request(
        sid,
        user_id=user_id,
        revoke_on_idle=True,
    )
    if error_code:
        logger.info(
            "institution_session_guard reject sid=%s error=%s",
            sid,
            error_code,
        )
        _mark_jwt_revocation_reason(error_code)
        if error_code == "idle_timeout":
            try:
                from ext import db

                db.session.commit()
            except Exception:
                from ext import db

                db.session.rollback()
        return True
    return False
