"""Tests gate — session resume et codes d'erreur (smoke)."""

from __future__ import annotations

from security.mobile_device_session_service import auth_capabilities


def test_auth_capabilities_advertise_session_resume():
    caps = auth_capabilities()
    assert caps["capabilities"]["session_resume"] is True
    assert caps["capabilities"]["idempotent_rotation"] is True
    assert caps["auth_contract_version"] == "mobile-device-session-v1"
