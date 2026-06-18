"""Tests unitaires refresh_token_service (device-aware, reuse same-device)."""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from models import User
from models.enums import UserRole
from security.refresh_token_service import (
    is_token_revoked,
    mark_token_rotated,
    revoke_active_tokens_for_device,
    store_refresh_token,
    update_token_last_used,
)


@pytest.fixture
def company_user(db_session):
    suffix = str(uuid.uuid4())[:8]
    user = User(
        username=f"rtd_{suffix}",
        email=f"rtd_{suffix}@test.local",
        public_id=str(uuid.uuid4()),
        role=UserRole.company,
    )
    user.set_password("password123", force_change=False)
    db_session.session.add(user)
    db_session.session.commit()
    return user


def test_revoke_active_tokens_for_device(app, company_user):
    with app.app_context():
        expires = datetime.now(UTC) + timedelta(days=30)
        token_a = f"token-a-{uuid.uuid4()}"
        token_b = f"token-b-{uuid.uuid4()}"
        store_refresh_token(token_a, company_user.id, expires, device_id="device-1")
        store_refresh_token(token_b, company_user.id, expires, device_id="device-2")

        revoked = revoke_active_tokens_for_device(
            company_user.id, "device-1", reason="test replace"
        )
        assert revoked == 1

        assert is_token_revoked(token_a) is True
        assert is_token_revoked(token_b) is False


def test_reuse_same_device_does_not_revoke_all(app, company_user, monkeypatch):
    with app.app_context():
        expires = datetime.now(UTC) + timedelta(days=30)
        old_token = f"old-refresh-{uuid.uuid4()}"
        new_token = f"new-refresh-{uuid.uuid4()}"

        store_refresh_token(
            old_token,
            company_user.id,
            expires,
            device_id="samsung-device-1",
        )
        store_refresh_token(
            new_token,
            company_user.id,
            expires,
            device_id="samsung-device-1",
        )
        mark_token_rotated(old_token, new_token)
        update_token_last_used(new_token)

        called = {"count": 0}

        def _fake_revoke_all(user_id: int, reason: str | None = None) -> int:
            called["count"] += 1
            return 0

        monkeypatch.setattr(
            "security.refresh_token_service.revoke_all_user_tokens",
            _fake_revoke_all,
        )

        rejected = is_token_revoked(
            old_token,
            request_device_id="samsung-device-1",
        )

        assert rejected is False
        assert called["count"] == 0


def test_reuse_different_device_still_rejected(app, company_user, monkeypatch):
    with app.app_context():
        expires = datetime.now(UTC) + timedelta(days=30)
        old_token = f"old-other-{uuid.uuid4()}"
        new_token = f"new-other-{uuid.uuid4()}"

        store_refresh_token(old_token, company_user.id, expires, device_id="device-a")
        store_refresh_token(new_token, company_user.id, expires, device_id="device-a")
        mark_token_rotated(old_token, new_token)
        update_token_last_used(new_token)

        called = {"count": 0}

        def _fake_revoke_all(user_id: int, reason: str | None = None) -> int:
            called["count"] += 1
            return 0

        monkeypatch.setattr(
            "security.refresh_token_service.revoke_all_user_tokens",
            _fake_revoke_all,
        )

        rejected = is_token_revoked(
            old_token,
            request_device_id="device-b",
        )

        assert rejected is True
        assert called["count"] == 0
