"""Tests F1a — logout preuves, revoke-pending idempotent, security revoke."""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

from ext import db
from models import User
from models.enums import UserRole
from models.mobile_device_session import MobileDeviceSessionStatus
from security import mobile_device_session_service as svc


@pytest.fixture
def session_user(db_session):
    suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"f1a_{suffix}",
        email=f"f1a_{suffix}@test.local",
        public_id=str(uuid.uuid4()),
        role=UserRole.driver,
    )
    user.set_password("password123", force_change=False)
    db_session.session.add(user)
    db_session.session.commit()
    return user


def test_revoke_pending_idempotent_replay(app, session_user):
    with app.app_context():
        session, _rec, secret = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"dev-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()
        op_id = str(uuid.uuid4())

        payload1, err1 = svc.revoke_pending_idempotent(
            session, secret, operation_id=op_id
        )
        assert err1 is None
        assert payload1["ok"] is True
        assert payload1["already_revoked"] is False
        db.session.commit()

        # Rejeu après perte ACK — même operation_id + même secret
        session = svc.get_session_by_id(session.session_id)
        payload2, err2 = svc.revoke_pending_idempotent(
            session, secret, operation_id=op_id
        )
        assert err2 is None
        assert payload2["ok"] is True
        assert payload2["already_revoked"] is True


def test_revoke_user_security_sessions_marks_security_revoked(app, session_user):
    with app.app_context():
        session, _rec, _sec = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"dev-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()
        tv_before = int(getattr(session_user, "token_version", 0) or 0)

        with patch(
            "security.refresh_token_service.revoke_all_user_tokens", return_value=0
        ):
            count = svc.revoke_user_security_sessions(
                session_user,
                reason="password_reset",
                increment_token_version=True,
            )
        db.session.commit()
        assert count == 1
        session = svc.get_session_by_id(session.session_id)
        assert session.status == MobileDeviceSessionStatus.security_revoked
        assert int(session_user.token_version) == tv_before + 1


def test_create_or_reuse_does_not_reactivate_terminal(app, session_user):
    with app.app_context():
        installation = f"dev-{uuid.uuid4()}"
        session1, _r1, _s1 = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=installation,
            role="driver",
        )
        db.session.commit()
        svc.revoke_session(
            session1,
            reason="test",
            status=MobileDeviceSessionStatus.security_revoked,
        )
        db.session.commit()
        old_id = session1.session_id

        session2, _r2, _s2 = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=installation,
            role="driver",
        )
        db.session.commit()
        # Nouvelle ligne (unicité partielle) — pas de réactivation
        assert session2.session_id != old_id
        assert session2.status == MobileDeviceSessionStatus.active
        old = svc.get_session_by_id(old_id)
        assert old.status == MobileDeviceSessionStatus.security_revoked


def test_logout_without_proof_rejected(client, app, session_user):
    with app.app_context():
        session, _r, _s = svc.create_or_reuse_session(
            user_id=session_user.id,
            device_installation_id=f"dev-{uuid.uuid4()}",
            role="driver",
        )
        db.session.commit()
        sid = str(session.session_id)

    resp = client.post(
        "/auth/logout",
        json={"session_id": sid},
        headers={"Content-Type": "application/json", "X-Client": "mobile"},
    )
    assert resp.status_code == 401
    body = resp.get_json() or {}
    assert body.get("error_code") == "logout_proof_required"

    with app.app_context():
        still = svc.get_session_by_id(sid)
        assert still is not None
        assert still.is_active()
