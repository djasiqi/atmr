"""Tests unitaires MobileDeviceSession / AuthRotationResult."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from models.mobile_device_session import (
    MobileDeviceSession,
    MobileDeviceSessionStatus,
)
from security import mobile_device_session_service as svc


def test_hash_credential_stable():
    assert svc.hash_credential("abc") == svc.hash_credential("abc")
    assert svc.hash_credential("abc") != svc.hash_credential("xyz")


def test_encrypt_decrypt_rotation_response_roundtrip():
    payload = {
        "access_token": "a" * 20,
        "refresh_token": "b" * 20,
        "session_id": str(uuid.uuid4()),
    }
    with patch.object(svc, "_get_encryption_key", return_value=(b"0" * 32, "v1")):
        ct, key_id = svc.encrypt_rotation_response(payload)
        assert key_id == "v1"
        assert isinstance(ct, (bytes, bytearray))
        restored = svc.decrypt_rotation_response(ct, key_id)
        assert restored["session_id"] == payload["session_id"]
        assert restored["access_token"] == payload["access_token"]


def test_validate_mobile_session_missing_id_is_legacy_ok():
    err, retryable = svc.validate_mobile_session(session_id=None, session_generation=None)
    assert err is None
    assert retryable is False


def test_validate_mobile_session_invalid_uuid():
    err, retryable = svc.validate_mobile_session(
        session_id="not-a-uuid", session_generation=1
    )
    assert err == "session_revoked"
    assert retryable is False


def test_device_session_limit_exception_payload():
    sessions = [MagicMock(serialize=lambda **_: {"session_id": "x"})]
    exc = svc.DeviceSessionLimitReached(sessions)
    assert str(exc) == "device_session_limit_reached"
    assert exc.sessions is sessions


def test_auth_capabilities_contract():
    caps = svc.auth_capabilities()
    assert caps["auth_contract_version"] == "mobile-device-session-v1"
    assert caps["capabilities"]["durable_device_session"] is True
    assert caps["capabilities"]["session_resume"] is True
