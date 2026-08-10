from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from models import DemoAccess, DemoRequest, User
from models.enums import UserRole
from services.demo.access_service import (
    DemoAccessError,
    consume_magic_link,
    enforce_demo_user_access_validity,
    expire_due_demo_accesses,
    provision_demo_access,
    resend_demo_access,
    revoke_demo_access,
)


@pytest.fixture(autouse=True)
def _isolate_demo_seed(monkeypatch):
    """Évite le seed lourd / les collisions inter-tests sur usernames démo."""
    monkeypatch.setattr(
        "services.demo.access_service.ensure_demo_reference_dataset",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "services.demo.access_service._ensure_demo_workspace_seeded",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "services.demo.access_service._reset_demo_dataset_on_session_start",
        lambda *args, **kwargs: None,
    )


def _demo_request() -> DemoRequest:
    suffix = uuid.uuid4().hex[:8]
    return DemoRequest(
        name="Demo User",
        email=f"demo.user.{suffix}@example.com",
        phone="+41790001122",
        organization="LIRIE Test Org",
        organization_type="clinic",
        use_case="planning_dispatch",
        volume_range="1_5",
        integration_required="no",
        integration_system=None,
        timing="immediate",
        preferred_slot="this_week",
        preferred_period="morning",
        comment="demo",
        score=12,
        status="new",
        trace_id="ct_demo_trace",
        source="web_demo_request",
        email_delivery_status="pending",
    )


def test_provision_and_consume_magic_link(db, monkeypatch):
    monkeypatch.setattr(
        "services.demo.access_service.send_demo_access_ready_email",
        lambda **kwargs: {"ok": True, "kwargs": kwargs},
    )
    request = _demo_request()
    db.session.add(request)
    db.session.commit()

    provision = provision_demo_access(demo_request_id=request.id)
    assert provision.demo_access.status == "active"
    assert provision.demo_access.demo_expires_at is not None
    assert provision.demo_access.magic_token_hash is not None

    consumed = consume_magic_link(provision.magic_token)
    assert consumed["ok"] is True
    assert consumed["session_created"] is True

    reused = consume_magic_link(provision.magic_token)
    assert reused["ok"] is True
    assert reused.get("already_consumed") is True


def test_provision_rejects_second_active_access(db, monkeypatch):
    monkeypatch.setattr(
        "services.demo.access_service.send_demo_access_ready_email",
        lambda **kwargs: {"ok": True, "kwargs": kwargs},
    )
    request = _demo_request()
    db.session.add(request)
    db.session.commit()

    provision_demo_access(demo_request_id=request.id)
    second = provision_demo_access(demo_request_id=request.id)
    assert second.reused_existing_access is True


def test_resend_rotates_token_without_extending_demo_expiry(db, monkeypatch):
    monkeypatch.setattr(
        "services.demo.access_service.send_demo_access_ready_email",
        lambda **kwargs: {"ok": True, "kwargs": kwargs},
    )
    request = _demo_request()
    db.session.add(request)
    db.session.commit()

    first = provision_demo_access(demo_request_id=request.id)
    original_access = first.demo_access
    original_demo_expiry = original_access.demo_expires_at
    original_token_hash = original_access.magic_token_hash

    resend = resend_demo_access(access_id=original_access.id)
    assert resend.demo_access.id == original_access.id
    assert resend.demo_access.magic_token_hash != original_token_hash
    assert resend.demo_access.demo_expires_at == original_demo_expiry


def test_revoke_blocks_consumption(db, monkeypatch):
    monkeypatch.setattr(
        "services.demo.access_service.send_demo_access_ready_email",
        lambda **kwargs: {"ok": True, "kwargs": kwargs},
    )
    request = _demo_request()
    db.session.add(request)
    db.session.commit()

    provision = provision_demo_access(demo_request_id=request.id)
    revoke_demo_access(access_id=provision.demo_access.id)

    with pytest.raises(DemoAccessError) as error:
        consume_magic_link(provision.magic_token)
    assert error.value.code in {"invalid_token", "access_revoked"}


def test_expire_due_accesses_marks_active_as_expired(db):
    request = _demo_request()
    db.session.add(request)
    db.session.flush()

    access = DemoAccess(
        demo_request_id=request.id,
        status="active",
        magic_token_hash="hash",
        magic_token_expires_at=datetime.now(UTC) + timedelta(minutes=10),
        demo_expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    db.session.add(access)
    db.session.commit()

    expired_count = expire_due_demo_accesses()
    db.session.refresh(access)
    assert expired_count >= 1
    assert access.status == "expired"
    assert access.magic_token_hash is None
    assert access.magic_token_expires_at is None


def test_consume_rejects_expired_token(db):
    request = _demo_request()
    db.session.add(request)
    db.session.flush()
    access = DemoAccess(
        demo_request_id=request.id,
        status="active",
        magic_token_hash=hashlib.sha256(b"expired-token").hexdigest(),
        magic_token_expires_at=datetime.now(UTC) - timedelta(seconds=1),
        demo_expires_at=datetime.now(UTC) + timedelta(hours=2),
    )
    db.session.add(access)
    db.session.commit()

    with pytest.raises(DemoAccessError) as error:
        consume_magic_link("expired-token")
    assert error.value.code == "token_expired"


def test_enforce_demo_user_access_validity_blocks_expired_demo_user(db):
    suffix = uuid.uuid4().hex[:8]
    user = User(
        username=f"demo_expired_user_{suffix}",
        email=f"demo-expired-{suffix}@example.com",
        role=UserRole.client,
        account_status="active",
    )
    user.set_password("Test1234!")
    db.session.add(user)
    db.session.flush()

    request = _demo_request()
    db.session.add(request)
    db.session.flush()

    access = DemoAccess(
        demo_request_id=request.id,
        demo_user_id=user.id,
        status="active",
        magic_token_hash="hash",
        magic_token_expires_at=datetime.now(UTC) + timedelta(minutes=10),
        demo_expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    db.session.add(access)
    db.session.commit()

    is_valid, message = enforce_demo_user_access_validity(user)
    db.session.refresh(user)
    db.session.refresh(access)

    assert is_valid is False
    assert message is not None
    assert access.status == "expired"
    assert user.account_status == "disabled"


def test_enforce_demo_user_access_validity_allows_active_demo_user(db):
    suffix = uuid.uuid4().hex[:8]
    user = User(
        username=f"demo_active_user_{suffix}",
        email=f"demo-active-{suffix}@example.com",
        role=UserRole.client,
        account_status="active",
    )
    user.set_password("Test1234!")
    db.session.add(user)
    db.session.flush()

    request = _demo_request()
    db.session.add(request)
    db.session.flush()

    access = DemoAccess(
        demo_request_id=request.id,
        demo_user_id=user.id,
        status="active",
        magic_token_hash="hash",
        magic_token_expires_at=datetime.now(UTC) + timedelta(minutes=10),
        demo_expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    db.session.add(access)
    db.session.commit()

    is_valid, message = enforce_demo_user_access_validity(user)
    db.session.refresh(user)
    db.session.refresh(access)

    assert is_valid is True
    assert message is None
    assert access.status == "active"
    assert user.account_status == "active"
