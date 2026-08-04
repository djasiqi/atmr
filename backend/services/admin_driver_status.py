"""Soft-disable / réactivation chauffeur admin + révocation sessions transactionnelle."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from ext import db
from models.driver import Driver
from models.enums import UserRole
from models.user import User
from security.audit_log import AuditLog
from services.admin_role_utils import normalized_role_value
from services.control_plane.projector import get_projector

logger = logging.getLogger(__name__)


class AdminDriverStatusError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error = error
        self.details = details or {}

    def to_response(self) -> tuple[dict[str, Any], int]:
        body: dict[str, Any] = {"message": self.message}
        if self.error:
            body["error"] = self.error
        if self.details:
            body["details"] = self.details
        return body, self.status_code


@dataclass
class DriverStatusResult:
    user_id: int
    driver_id: int
    is_active: bool
    is_available: bool
    sessions_revoked: int
    reauthentication_required: bool
    status: str = "updated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "user_id": self.user_id,
            "driver_id": self.driver_id,
            "is_active": self.is_active,
            "is_available": self.is_available,
            "sessions_revoked": self.sessions_revoked,
            "reauthentication_required": self.reauthentication_required,
        }


@dataclass
class RevokeSessionsResult:
    user_id: int
    mobile_sessions_revoked: int
    token_version_incremented: bool
    reauthentication_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "mobile_sessions_revoked": self.mobile_sessions_revoked,
            "token_version_incremented": self.token_version_incremented,
            "reauthentication_required": self.reauthentication_required,
        }


def _now() -> datetime:
    return datetime.now(UTC)


def set_driver_status(
    *,
    user_id: int,
    is_active: bool,
    reason: str,
    actor_admin_id: int,
    expected_is_active: bool | None = None,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> DriverStatusResult:
    if not reason or len(reason.strip()) < 5:
        raise AdminDriverStatusError(
            "Une raison d'au moins 5 caractères est requise.",
            status_code=400,
            error="reason_required",
        )

    user = db.session.execute(
        select(User).where(User.id == user_id).with_for_update()
    ).scalar_one_or_none()
    if user is None:
        raise AdminDriverStatusError(
            "Compte introuvable.", status_code=404, error="user_not_found"
        )

    if normalized_role_value(user.role) != UserRole.DRIVER.value:
        raise AdminDriverStatusError(
            "Le compte n'est pas un chauffeur.",
            status_code=409,
            error="not_a_driver",
        )

    driver = db.session.execute(
        select(Driver).where(Driver.user_id == user_id).with_for_update()
    ).scalar_one_or_none()
    if driver is None:
        raise AdminDriverStatusError(
            "Profil chauffeur introuvable.",
            status_code=404,
            error="driver_not_found",
        )

    current_active = bool(driver.is_active)
    if expected_is_active is not None and bool(expected_is_active) != current_active:
        raise AdminDriverStatusError(
            "L'état du chauffeur a changé concurrentement.",
            status_code=409,
            error="driver_status_changed",
            details={
                "expected_is_active": expected_is_active,
                "current_is_active": current_active,
            },
        )

    if current_active == bool(is_active):
        return DriverStatusResult(
            user_id=int(user.id),
            driver_id=int(driver.id),
            is_active=current_active,
            is_available=bool(driver.is_available),
            sessions_revoked=0,
            reauthentication_required=False,
            status="unchanged",
        )

    old_is_active = current_active
    old_is_available = bool(driver.is_available)
    sessions_revoked = 0
    action_type = (
        "admin_driver_disabled" if not is_active else "admin_driver_reactivated"
    )

    try:
        # Savepoint : échec revoke / sync n'empoisonne pas la session appelante
        with db.session.begin_nested():
            driver.is_active = bool(is_active)
            if not is_active:
                driver.is_available = False
            db.session.flush()

            if not is_active:
                from security.mobile_device_session_service import (
                    revoke_user_security_sessions,
                )

                sessions_revoked = revoke_user_security_sessions(
                    user,
                    reason="admin_driver_disabled",
                    increment_token_version=True,
                    fail_closed=True,
                    commit_tokens=False,
                )

            get_projector().sync_driver(driver)
            db.session.flush()

            audit = AuditLog()
            audit.user_id = actor_admin_id
            audit.user_type = "ADMIN"
            audit.action_type = action_type
            audit.action_category = "security"
            audit.action_details = json.dumps(
                {
                    "target_user_id": user.id,
                    "driver_id": driver.id,
                    "old_is_active": old_is_active,
                    "new_is_active": bool(driver.is_active),
                    "old_is_available": old_is_available,
                    "new_is_available": bool(driver.is_available),
                    "reason": reason.strip(),
                    "sessions_revoked": sessions_revoked,
                },
                ensure_ascii=False,
                default=str,
            )
            audit.result_status = "success"
            audit.result_message = action_type
            audit.ip_address = ip_address
            audit.user_agent = user_agent
            audit.company_id = driver.company_id
            audit.driver_id = driver.id
            audit.resource_type = "driver"
            audit.resource_id = str(driver.id)
            audit.created_at = _now()
            db.session.add(audit)
    except AdminDriverStatusError:
        raise
    except Exception as exc:
        raise AdminDriverStatusError(
            "Impossible de révoquer les sessions ; désactivation annulée."
            if not is_active
            else "Échec de la mise à jour du statut chauffeur.",
            status_code=503,
            error="session_revoke_failed" if not is_active else "driver_status_failed",
            details={"detail": str(exc)},
        ) from exc

    db.session.commit()
    return DriverStatusResult(
        user_id=int(user.id),
        driver_id=int(driver.id),
        is_active=bool(driver.is_active),
        is_available=bool(driver.is_available),
        sessions_revoked=int(sessions_revoked),
        reauthentication_required=not is_active,
        status="updated",
    )


def revoke_user_sessions_admin(
    *,
    user_id: int,
    reason: str,
    actor_admin_id: int,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> RevokeSessionsResult:
    if not reason or len(reason.strip()) < 5:
        raise AdminDriverStatusError(
            "Une raison d'au moins 5 caractères est requise.",
            status_code=400,
            error="reason_required",
        )

    user = db.session.execute(
        select(User).where(User.id == user_id).with_for_update()
    ).scalar_one_or_none()
    if user is None:
        raise AdminDriverStatusError(
            "Compte introuvable.", status_code=404, error="user_not_found"
        )

    from security.mobile_device_session_service import revoke_user_security_sessions

    previous_version = int(getattr(user, "token_version", 0) or 0)
    count = 0
    try:
        with db.session.begin_nested():
            count = revoke_user_security_sessions(
                user,
                reason="admin_user_sessions_revoked",
                increment_token_version=True,
                fail_closed=True,
                commit_tokens=False,
            )
            audit = AuditLog()
            audit.user_id = actor_admin_id
            audit.user_type = "ADMIN"
            audit.action_type = "admin_user_sessions_revoked"
            audit.action_category = "security"
            audit.action_details = json.dumps(
                {
                    "target_user_id": user.id,
                    "reason": reason.strip(),
                    "mobile_sessions_revoked": count,
                    "token_version_before": previous_version,
                    "token_version_after": int(getattr(user, "token_version", 0) or 0),
                },
                ensure_ascii=False,
                default=str,
            )
            audit.result_status = "success"
            audit.result_message = "sessions_revoked"
            audit.ip_address = ip_address
            audit.user_agent = user_agent
            audit.resource_type = "user"
            audit.resource_id = str(user.id)
            audit.created_at = _now()
            db.session.add(audit)
    except Exception as exc:
        raise AdminDriverStatusError(
            "Impossible de révoquer les sessions.",
            status_code=503,
            error="session_revoke_failed",
            details={"detail": str(exc)},
        ) from exc

    db.session.commit()

    return RevokeSessionsResult(
        user_id=int(user.id),
        mobile_sessions_revoked=int(count),
        token_version_incremented=True,
        reauthentication_required=True,
    )
