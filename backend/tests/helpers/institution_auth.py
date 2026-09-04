"""Helpers JWT institution avec WebSession (garde sid P0-A)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from flask_jwt_extended import create_access_token

from models import Institution, User
from models.enums import UserRole
from models.web_session import WebSession


def institution_bearer_headers(
    db,
    user: User,
    institution: Institution,
    *,
    institution_role: str | None = None,
    extra_claims: dict[str, Any] | None = None,
) -> dict[str, str]:
    """JWT institution valide : crée une WebSession et injecte ``sid``."""
    now = datetime.now(UTC)
    session = WebSession()
    session.id = str(uuid.uuid4())
    session.user_id = int(user.id)
    session.institution_id = institution.id
    session.created_at = now
    session.expires_at = now + timedelta(hours=8)
    session.last_interactive_activity_at = now
    db.session.add(session)
    db.session.flush()

    role = (
        institution_role
        or getattr(user, "institution_role", None)
        or "institution_admin"
    )
    claims: dict[str, Any] = {
        "role": UserRole.INSTITUTION.value
        if hasattr(UserRole, "INSTITUTION")
        else str(getattr(user.role, "value", user.role)),
        "institution_id": institution.id,
        "institution_role": role,
        "sid": session.id,
        "aud": "atmr-api",
    }
    if extra_claims:
        claims.update(extra_claims)

    token = create_access_token(
        identity=str(user.public_id),
        additional_claims=claims,
    )
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
