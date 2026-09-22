"""Contrat AUTH-SMS-02 : e-mail active le compte, SMS pour le 1er transport."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

import services.auth.portal_phone_verification as portal_mod
from models.enums import ClientType, UserRole
from services.auth.portal_phone_verification import (
    PortalPhoneVerificationRequired,
    apply_user_phone_change,
    assert_portal_can_confirm_transport,
    is_portal_client_user,
    maybe_promote_portal_account,
    promote_portal_account_after_email,
    user_phone_is_verified,
)
from services.notifications.phone_e164 import normalize_e164_phone


def test_is_portal_client_user_only_portal():
    portal = SimpleNamespace(
        role=UserRole.CLIENT,
        clients=[SimpleNamespace(client_type=ClientType.PORTAL)],
    )
    institution_like = SimpleNamespace(
        role=UserRole.CLIENT,
        clients=[SimpleNamespace(client_type=None)],
    )
    driver = SimpleNamespace(role=UserRole.DRIVER, clients=[])
    assert is_portal_client_user(portal) is True
    assert is_portal_client_user(institution_like) is False
    assert is_portal_client_user(driver) is False


def test_promote_requires_email_and_keeps_phone_unverified():
    session = SimpleNamespace(
        email_verified_at=datetime.now(UTC),
        consumed_at=None,
        activation_session_id="s1",
    )
    user = SimpleNamespace(
        id=7,
        role=UserRole.CLIENT,
        account_status="pending_activation",
        phone_verified_at=None,
        clients=[SimpleNamespace(client_type=ClientType.PORTAL, is_active=False)],
    )
    assert promote_portal_account_after_email(user, session) is True
    assert user.account_status == "active"
    assert user.clients[0].is_active is True
    assert session.consumed_at is not None
    assert user.phone_verified_at is None


def test_phone_change_revokes_only_when_normalized_differs():
    user = SimpleNamespace(
        id=1, phone="+41768190077", phone_verified_at=datetime.now(UTC)
    )
    apply_user_phone_change(user, "076 819 00 77")
    assert user.phone == "+41768190077"
    assert user.phone_verified_at is not None

    apply_user_phone_change(user, "+41791230000")
    assert user.phone == "+41791230000"
    assert user.phone_verified_at is None


def test_maybe_promote_skips_non_portal_and_unverified_email(monkeypatch):
    institution = SimpleNamespace(
        role=UserRole.INSTITUTION,
        account_status="pending_activation",
        clients=[],
    )
    assert maybe_promote_portal_account(institution) is False

    portal_unverified = SimpleNamespace(
        id=3,
        email="portal@example.test",
        role=UserRole.CLIENT,
        account_status="pending_activation",
        clients=[SimpleNamespace(client_type=ClientType.PORTAL, is_active=False)],
    )
    monkeypatch.setattr(portal_mod, "latest_activation_session", lambda _user: None)
    assert maybe_promote_portal_account(portal_unverified) is False

    session = SimpleNamespace(
        email_verified_at=datetime.now(UTC),
        consumed_at=None,
        activation_session_id="s-promo",
    )
    monkeypatch.setattr(portal_mod, "latest_activation_session", lambda _user: session)
    assert maybe_promote_portal_account(portal_unverified) is True
    assert portal_unverified.account_status == "active"
    assert portal_unverified.clients[0].is_active is True


def test_assert_portal_can_confirm_skips_transport_and_blocks_unverified():
    assert_portal_can_confirm_transport(
        user_id=1,
        client=SimpleNamespace(client_type=ClientType.TRANSPORT),
        user_loader=lambda _uid: SimpleNamespace(phone_verified_at=None),
    )
    assert_portal_can_confirm_transport(
        user_id=1,
        client=SimpleNamespace(client_type=ClientType.PORTAL),
        user_loader=lambda _uid: SimpleNamespace(phone_verified_at=datetime.now(UTC)),
    )
    with pytest.raises(PortalPhoneVerificationRequired):
        assert_portal_can_confirm_transport(
            user_id=1,
            client=SimpleNamespace(client_type=ClientType.PORTAL),
            user_loader=lambda _uid: SimpleNamespace(phone_verified_at=None),
        )


def test_user_phone_is_verified_uses_timestamp_only():
    assert (
        user_phone_is_verified(
            SimpleNamespace(phone="+41768190077", phone_verified_at=None)
        )
        is False
    )
    assert (
        user_phone_is_verified(
            SimpleNamespace(phone=None, phone_verified_at=datetime.now(UTC))
        )
        is True
    )
    assert normalize_e164_phone("0768190077") == "+41768190077"
