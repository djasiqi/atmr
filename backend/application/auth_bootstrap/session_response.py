"""Construction du JSON `CurrentSessionResponse` (shape minimal contractuel)."""

from __future__ import annotations

from typing import Any

from models.enums import UserRole

from . import access_denied_codes as codes
from .session_bootstrap_snapshot import SessionBootstrapSnapshot

BOOTSTRAP_RESPONSE_VERSION = 1


def _company_id_value(snapshot: SessionBootstrapSnapshot) -> int | None:
    if snapshot.driver_company_id is not None:
        return snapshot.driver_company_id
    return snapshot.company_relation_id


def _profile_type(role: UserRole) -> str | None:
    if role == UserRole.DRIVER:
        return "driver"
    if role == UserRole.CLIENT:
        return "client"
    if role == UserRole.INSTITUTION:
        return "institution"
    return None


def _profile_active(snapshot: SessionBootstrapSnapshot) -> bool | None:
    if snapshot.role == UserRole.DRIVER:
        if snapshot.driver_id is None:
            return None
        return bool(snapshot.driver_is_active)
    if snapshot.role == UserRole.CLIENT:
        if not snapshot.client_active_flags:
            return None
        return any(snapshot.client_active_flags)
    return None


def _legacy_reason(access_code: str) -> str:
    if access_code == codes.PENDING_ACTIVATION:
        return "account_pending_activation"
    return "account_disabled"


def build_auth_me_payload(
    snapshot: SessionBootstrapSnapshot,
    denial: tuple[str, str] | None,
) -> dict[str, Any]:
    """Payload GET /auth/me — toutes les clés contractuelles présentes."""
    company_id = _company_id_value(snapshot)
    profile_type = _profile_type(snapshot.role)
    profile_active = _profile_active(snapshot)

    base: dict[str, Any] = {
        "id": snapshot.user_id,
        "public_id": snapshot.public_id,
        "username": snapshot.username,
        "email": snapshot.email,
        "role": snapshot.role.value,
        "bootstrap_version": BOOTSTRAP_RESPONSE_VERSION,
        "account_active": denial is None,
        "profile_active": profile_active,
        "profile_type": profile_type,
        "company_id": company_id,
        "driver_id": snapshot.driver_id,
        "access_denied_code": denial[0] if denial else None,
        "message": denial[1] if denial else None,
    }

    if denial is not None:
        code, msg = denial
        base["error"] = msg
        base["reason"] = _legacy_reason(code)
    return base
