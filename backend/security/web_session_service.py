"""Service de session web durable (claim JWT sid) — inactivité humaine institution."""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from flask import current_app, request

from ext import db
from models.enums import UserRole
from models.web_session import WebSession

if TYPE_CHECKING:
    from models.user import User

logger = logging.getLogger(__name__)

JWT_SID_CLAIM = "sid"
SESSION_UPGRADE_REQUIRED = "session_upgrade_required"


def institution_idle_timeout_seconds() -> int:
    return int(current_app.config.get("INSTITUTION_IDLE_TIMEOUT_SECONDS", 900))


def is_institution_user(user: User | None) -> bool:
    if user is None:
        return False
    role = getattr(user, "role", None)
    role_value = role.value if hasattr(role, "value") else str(role or "")
    if role_value.upper() == UserRole.INSTITUTION.value:
        return True
    return getattr(user, "institution_id", None) is not None


def extract_sid_from_claims(claims: dict | None) -> str | None:
    if not claims:
        return None
    raw = claims.get(JWT_SID_CLAIM) or claims.get("sid")
    if raw is None:
        return None
    value = str(raw).strip()
    return value or None


def is_institution_jwt_claims(claims: dict | None) -> bool:
    """True si le JWT représente un compte institution (rôle ou claim institution_id)."""
    if not claims:
        return False
    role = str(claims.get("role") or "").upper()
    if role == UserRole.INSTITUTION.value:
        return True
    return claims.get("institution_id") is not None


def claims_require_institution_session_guard(claims: dict | None) -> bool:
    """True si le JWT institution doit passer par la garde session (P0-A.1)."""
    return is_institution_jwt_claims(claims)


def get_web_session_by_id(
    session_id: str | None,
    *,
    for_update: bool = False,
) -> WebSession | None:
    if not session_id:
        return None
    query = WebSession.query.filter_by(id=str(session_id))
    if for_update:
        query = query.with_for_update()
    return query.first()


def create_web_session(
    user: User,
    *,
    expires_at: datetime,
    remember_me: bool = False,
) -> WebSession:
    """Crée une session web stable pour login institution."""
    now = datetime.now(UTC)
    session = WebSession(
        id=str(uuid.uuid4()),
        user_id=user.id,
        institution_id=getattr(user, "institution_id", None),
        created_at=now,
        expires_at=expires_at,
        last_interactive_activity_at=now,
        ip_address=(request.remote_addr if request else None),
        user_agent=(request.headers.get("User-Agent") if request else None),
    )
    db.session.add(session)
    logger.info(
        "web_session_created sid=%s user_id=%s remember_me=%s",
        session.id,
        user.id,
        remember_me,
    )
    return session


def record_interactive_activity(
    session_id: str | None,
    *,
    user_id: int | None = None,
    min_interval_seconds: int = 30,
) -> tuple[bool, str | None]:
    """Heartbeat humain — met à jour last_interactive_activity_at.

    Returns:
        (updated, error_code)
    """
    session = get_web_session_by_id(session_id, for_update=True)
    if session is None:
        return False, "session_not_found"
    if user_id is not None and session.user_id != user_id:
        return False, "session_user_mismatch"
    if session.is_revoked():
        return False, "session_revoked"
    if not session.is_active():
        return False, "session_expired"

    now = datetime.now(UTC)
    last = session.last_interactive_activity_at
    if last is not None:
        delta = (now - last).total_seconds()
        if delta < min_interval_seconds:
            return False, None

    session.last_interactive_activity_at = now
    return True, None


def validate_web_session_for_request(
    session_id: str | None,
    *,
    user_id: int | None = None,
    revoke_on_idle: bool = True,
) -> str | None:
    """Valide sid pour une requête institution. Ne met jamais à jour l'activité.

    Returns:
        error_code ou None si OK
    """
    session = get_web_session_by_id(session_id)
    if session is None:
        return "session_not_found"
    if user_id is not None and session.user_id != user_id:
        return "session_user_mismatch"
    if session.is_revoked():
        return "session_revoked"
    if not session.is_active():
        return "session_expired"

    idle_limit = institution_idle_timeout_seconds()
    reference = session.last_interactive_activity_at or session.created_at
    now = datetime.now(UTC)
    idle_seconds = (now - reference).total_seconds()
    if idle_seconds >= idle_limit:
        if revoke_on_idle:
            revoke_web_session(
                session.id,
                reason="idle_timeout",
                commit=False,
            )
        return "idle_timeout"

    return None


def revoke_web_session(
    session_id: str | None,
    *,
    reason: str = "logout",
    commit: bool = True,
) -> bool:
    session = get_web_session_by_id(session_id, for_update=True)
    if session is None:
        return False
    if session.is_revoked():
        return True
    now = datetime.now(UTC)
    session.revoked_at = now
    session.revoked_reason = reason[:255] if reason else "logout"

    from security.refresh_token_service import revoke_refresh_tokens_for_web_session

    revoke_refresh_tokens_for_web_session(session_id, reason=reason, commit=False)

    if commit:
        db.session.commit()
    return True


def revoke_all_user_web_sessions(
    user_id: int,
    *,
    reason: str = "logout",
    commit: bool = True,
) -> int:
    sessions = (
        WebSession.query.filter(
            WebSession.user_id == user_id,
            WebSession.revoked_at.is_(None),
        )
        .with_for_update()
        .all()
    )
    count = 0
    for session in sessions:
        if revoke_web_session(session.id, reason=reason, commit=False):
            count += 1
    if commit and count:
        db.session.commit()
    return count


def resolve_web_session_expires(
    *,
    remember_me: bool,  # noqa: ARG001 — conservé pour contrat login institution
    refresh_expires_delta: timedelta,
) -> datetime:
    return datetime.now(UTC) + refresh_expires_delta
