"""Helper central pour les audit logs.

Extrait automatiquement user/company/IP du contexte Flask.
Utilise flask.g pour eviter les N+1 queries (G8).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def audit_log(
    action_type: str,
    category: str,
    resource_type: str | None = None,
    resource_id: str | int | None = None,
    result: str = "success",
    result_message: str | None = None,
    user: Any = None,
    company: Any = None,
    action_details: dict[str, Any] | None = None,
    **kwargs: Any,
) -> None:
    """Enregistre un audit log en extrayant le contexte Flask automatiquement.

    Priorite de resolution user/company :
    1. Parametres explicites user= / company=
    2. flask.g.current_user / flask.g.current_company (set par get_company_from_token)
    3. Fallback via JWT identity (query DB)
    """
    from flask import g, has_request_context, request

    from security.audit_log import AuditLogger

    _user = user or getattr(g, "current_user", None) if has_request_context() else user
    _company = (
        company or getattr(g, "current_company", None)
        if has_request_context()
        else company
    )

    if not _user and has_request_context():
        try:
            from flask_jwt_extended import get_jwt_identity

            from models import User

            pid = get_jwt_identity()
            if pid:
                _user = User.query.filter_by(public_id=pid).first()
        except Exception:
            pass

    if not _company and _user:
        _company = getattr(_user, "company", None)

    ip_address = None
    user_agent_str = None
    if has_request_context():
        ip_address = request.remote_addr
        user_agent_str = request.headers.get("User-Agent")

    try:
        AuditLogger.log_action(
            action_type=action_type,
            action_category=category,
            user_id=_user.id if _user else None,
            user_type=_user.role.value if _user and hasattr(_user, "role") and _user.role else "system",
            company_id=_company.id if _company else None,
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id else None,
            result_status=result,
            result_message=result_message,
            action_details=action_details,
            ip_address=ip_address,
            user_agent=user_agent_str,
            **kwargs,
        )
    except Exception:
        logger.exception("[audit_log] Failed to write audit log for %s", action_type)
