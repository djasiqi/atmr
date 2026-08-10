"""Tests unitaires MobileDeviceSession / AuthRotationResult."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from ext import db
from models import User
from models.enums import UserRole
from models.mobile_device_session import (
    MobileDeviceSession,
    MobileDeviceSessionStatus,
)
from security import mobile_device_session_service as svc


@pytest.fixture
def session_user(db_session):
    """Utilisateur chauffeur de test pour les sessions durables mobile."""
    suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"mds_{suffix}",
        email=f"mds_{suffix}@test.local",
        public_id=str(uuid.uuid4()),
        role=UserRole.driver,
    )
    user.set_password("password123", force_change=False)
    db_session.session.add(user)
    db_session.session.commit()
    return user


def test_hash_credential_stable():
    assert svc.hash_credential("abc") == svc.hash_credential("abc")
    assert svc.hash_credential("abc") != svc.hash_credential("xyz")


def test_encrypt_decrypt_rotation_response_roundtrip():
    payload = {
        "access_token": "a" * 20,
        "refresh_token": "b" * 20,
        "session_id": str(uuid.uuid4()),
    }
    with patch.object(
        svc, "_get_encryption_key_for_id", return_value=(b"0" * 32, "v1")
    ):
        ct, key_id = svc.encrypt_rotation_response(payload)
        assert key_id == "v1"
        assert isinstance(ct, (bytes, bytearray))
        restored = svc.decrypt_rotation_response(ct, key_id)
        assert restored["session_id"] == payload["session_id"]
        assert restored["access_token"] == payload["access_token"]


def test_validate_mobile_session_missing_id_is_legacy_ok():
    err, retryable = svc.validate_mobile_session(
        session_id=None, session_generation=None
    )
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


def test_create_or_reuse_session_creates_new(app, session_user):
    with app.app_context():
        session, recovery, revocation = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"device-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()

        assert session.user_id == session_user.id
        assert session.generation == 1
        assert session.status == MobileDeviceSessionStatus.active
        assert svc.verify_recovery_credential(session, recovery) is True
        assert svc.verify_revocation_secret(session, revocation) is True


def test_create_or_reuse_session_reuses_same_installation(app, session_user):
    with app.app_context():
        installation_id = f"device-{uuid.uuid4()}"

        session1, recovery1, _revocation1 = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=installation_id,
            role="driver",
        )
        db.session.commit()

        session2, recovery2, _revocation2 = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=installation_id,
            role="driver",
        )
        db.session.commit()

        # Même installation ⇒ même session_id, generation incrémentée.
        assert session1.session_id == session2.session_id
        assert session2.generation == 2
        assert recovery1 != recovery2

        # Ancien credential encore valide pendant la fenêtre de grâce.
        assert svc.verify_recovery_credential(session2, recovery1) is True
        assert svc.verify_recovery_credential(session2, recovery2) is True


def test_create_or_reuse_session_limit_reached(app, session_user, monkeypatch):
    monkeypatch.setenv("MAX_MOBILE_DEVICE_SESSIONS_DRIVER", "1")
    with app.app_context():
        svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"device-a-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()

        with pytest.raises(svc.DeviceSessionLimitReached) as exc_info:
            svc.create_or_reuse_session(
                user_id=session_user.id,
                device_installation_id=f"device-b-{uuid.uuid4()}",
                role="driver",
            )

        assert len(exc_info.value.sessions) == 1


def test_validate_mobile_session_active_session_ok(app, session_user):
    with app.app_context():
        session, _recovery, _revocation = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"device-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()

        err, retryable = svc.validate_mobile_session(
            session_id=str(session.session_id),
            session_generation=session.generation,
            user_id=session_user.id,
        )
        assert err is None
        assert retryable is False


def test_validate_mobile_session_generation_mismatch(app):
    with app.app_context():
        err, retryable = svc.validate_mobile_session(
            session_id=str(uuid.uuid4()),
            session_generation=99,
            user_id=1,
        )
    assert err == "session_revoked"
    assert retryable is False


def test_validate_mobile_session_after_revocation(app, session_user):
    with app.app_context():
        session, _recovery, _revocation = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"device-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()

        svc.revoke_session(session, reason="test_revocation")
        db.session.commit()

        err, retryable = svc.validate_mobile_session(
            session_id=str(session.session_id),
            session_generation=session.generation,
            user_id=session_user.id,
        )
        assert err == "session_revoked"
        assert retryable is False


def test_consume_revocation_secret_revokes_session(app, session_user):
    with app.app_context():
        session, _recovery, revocation = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"device-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()

        assert svc.consume_revocation_secret(session, revocation) is True
        db.session.commit()

        assert session.is_active() is False
        # Le secret est one-shot : un second appel échoue.
        assert svc.consume_revocation_secret(session, revocation) is False


def test_store_and_load_rotation_result_roundtrip(app, session_user):
    with app.app_context():
        session, _recovery, _revocation = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"device-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()

        payload = {"access_token": "a" * 10, "refresh_token": "b" * 10}
        with patch.object(
            svc, "_get_encryption_key_for_id", return_value=(b"1" * 32, "v1")
        ):
            row = svc.store_rotation_result(
                session=session,
                idempotency_key="idem-key-1",
                request_generation=session.generation,
                successor_generation=session.generation,
                response_payload=payload,
            )
            db.session.commit()

            fetched = svc.get_rotation_result(session.session_id, "idem-key-1")
            assert fetched is not None
            assert row is not None
            assert fetched.id == row.id

            restored = svc.load_rotation_response(fetched)
            assert restored == payload
