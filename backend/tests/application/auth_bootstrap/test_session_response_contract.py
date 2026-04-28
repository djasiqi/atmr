"""Contrat JSON GET /auth/me."""

from __future__ import annotations

from application.auth_bootstrap import access_denied_codes as codes
from application.auth_bootstrap.session_bootstrap_snapshot import (
    SessionBootstrapSnapshot,
)
from application.auth_bootstrap.session_response import build_auth_me_payload
from models.enums import UserRole

REQUIRED_KEYS = frozenset(
    {
        "id",
        "public_id",
        "username",
        "email",
        "role",
        "bootstrap_version",
        "account_active",
        "profile_active",
        "profile_type",
        "company_id",
        "driver_id",
        "access_denied_code",
        "message",
    }
)


def _base_snap() -> SessionBootstrapSnapshot:
    return SessionBootstrapSnapshot(
        user_id=42,
        public_id="p-1",
        username="john",
        email="j@example.com",
        role=UserRole.DRIVER,
        account_status=None,
        driver_id=7,
        driver_company_id=3,
        driver_is_active=True,
        company_relation_id=None,
        client_active_flags=(),
    )


def test_success_payload_has_all_keys():
    p = build_auth_me_payload(_base_snap(), None)
    assert REQUIRED_KEYS.issubset(p.keys())
    assert p["access_denied_code"] is None
    assert p["message"] is None
    assert p["account_active"] is True
    assert "error" not in p


def test_forbidden_payload_has_error_and_reason():
    p = build_auth_me_payload(
        _base_snap(),
        (codes.DRIVER_PROFILE_INACTIVE, "Compte désactivé"),
    )
    assert REQUIRED_KEYS.issubset(p.keys())
    assert p["access_denied_code"] == codes.DRIVER_PROFILE_INACTIVE
    assert p["message"] == "Compte désactivé"
    assert p["error"] == "Compte désactivé"
    assert p["reason"] == "account_disabled"
