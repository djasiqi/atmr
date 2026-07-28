"""F-03 — supersession des liens d'activation email."""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from models.activation_email_delivery import (
    EMAIL_DELIVERY_KIND_INITIAL,
    EMAIL_DELIVERY_KIND_RESEND,
    EMAIL_DELIVERY_QUEUED,
    EMAIL_DELIVERY_SENT,
    ActivationEmailDelivery,
)
from models.activation_session import ActivationSession
from services.notifications.activation_email_delivery import (
    can_start_new_delivery_snapshot,
    cas_claim_sending,
    finalize_after_provider_accepted,
    mark_delivery_failed,
    resolve_activation_token_for_delivery,
    set_current_delivery,
    sync_current_delivery_mirror,
    try_enqueue_activation_email,
)
from services.notifications.activation_token import (
    derive_activation_token,
    hash_activation_token,
)
from services.security.activation_legacy import (
    ActivationLegacyConfigError,
    get_legacy_acceptance_window,
    is_legacy_acceptance_active,
    validate_activation_legacy_for_boot,
)

VERIFY_PATH = "/api/v1/auth/activation/verify-email"


@pytest.fixture(autouse=True)
def _activation_key(monkeypatch):
    monkeypatch.setenv("ACTIVATION_TOKEN_KEY_V1", "test-activation-key-v1")
    monkeypatch.delenv("ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_FROM_UTC", raising=False)
    monkeypatch.delenv("ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_UNTIL_UTC", raising=False)


def _session(db) -> ActivationSession:
    s = ActivationSession()
    s.activation_session_id = str(uuid.uuid4())
    s.user_id = 1
    s.sms_attempts = 0
    s.resend_count_email = 0
    s.resend_count_sms = 0
    db.session.add(s)
    db.session.commit()
    return s


def _delivery(
    db,
    session: ActivationSession,
    *,
    expires: datetime | None = None,
    superseded: bool = False,
    status: str = EMAIL_DELIVERY_QUEUED,
    kind: str = EMAIL_DELIVERY_KIND_INITIAL,
    make_current: bool = True,
) -> tuple[ActivationEmailDelivery, str]:
    did = str(uuid.uuid4())
    token = derive_activation_token(did, key_version=1)
    delivery = ActivationEmailDelivery(
        activation_session_pk=session.id,
        email_delivery_id=did,
        kind=kind,
        status=status,
        token_key_version=1,
        email_token_hash=hash_activation_token(token),
        token_expires_at=expires or (datetime.now(UTC) + timedelta(minutes=30)),
        superseded_at=datetime.now(UTC) if superseded else None,
    )
    db.session.add(delivery)
    if make_current and not superseded:
        set_current_delivery(session, delivery)
    db.session.commit()
    return delivery, token


class TestVerifySupersession:
    def test_expired_delivery_future_session_mirror_rejected(self, client, db):
        session = _session(db)
        _delivery_row, token = _delivery(
            db,
            session,
            expires=datetime.now(UTC) - timedelta(minutes=1),
        )
        # Miroir futur (bug pré-F-03)
        session.email_token_expires_at = datetime.now(UTC) + timedelta(hours=1)
        db.session.commit()

        resp = client.post(VERIFY_PATH, json={"token": token})
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "token_expired"

    def test_future_delivery_expired_session_mirror_ok(self, client, db):
        session = _session(db)
        _delivery_row, token = _delivery(db, session)
        session.email_token_expires_at = datetime.now(UTC) - timedelta(hours=1)
        db.session.commit()

        resp = client.post(VERIFY_PATH, json={"token": token})
        assert resp.status_code == 200
        assert "confirm" in (resp.get_json().get("message") or "").lower()

    def test_resend_invalidates_previous_token(self, client, db, monkeypatch):
        session = _session(db)
        # Livraison A déjà « sent » (sinon snapshot bloque queued/sending)
        _a, token_a = _delivery(db, session, status=EMAIL_DELIVERY_SENT)
        session.last_email_sent_at = datetime.now(UTC) - timedelta(hours=1)
        db.session.commit()

        monkeypatch.setattr(
            "services.notifications.activation_email_delivery.is_email_provider_configured",
            lambda: (True, None),
        )
        monkeypatch.setattr(
            "services.notifications.activation_email_delivery.enqueue_activation_email",
            lambda **kwargs: None,
        )
        result = try_enqueue_activation_email(
            session,
            kind=EMAIL_DELIVERY_KIND_RESEND,
            environment="testing",
            is_testing=True,
        )
        assert result["ok"] is True
        token_b = result["email_token"]
        assert token_b

        resp_a = client.post(VERIFY_PATH, json={"token": token_a})
        assert resp_a.status_code == 400
        body_a = resp_a.get_json() or {}
        assert body_a.get("error") == "token_expired"
        assert "remplacé" in (body_a.get("message") or "").lower()

        resp_b = client.post(VERIFY_PATH, json={"token": token_b})
        assert resp_b.status_code == 200

    def test_token_expires_at_null_invalid(self, client, db):
        session = _session(db)
        delivery, token = _delivery(db, session)
        delivery.token_expires_at = None
        db.session.commit()
        resp = client.post(VERIFY_PATH, json={"token": token})
        assert resp.status_code == 400
        assert resp.get_json()["error"] == "token_invalid"

    def test_already_confirmed_via_b_rejects_a(self, client, db):
        session = _session(db)
        a, token_a = _delivery(db, session)
        a.superseded_at = datetime.now(UTC)
        b, token_b = _delivery(db, session, kind=EMAIL_DELIVERY_KIND_RESEND)
        a.superseded_by_delivery_id = b.email_delivery_id
        db.session.commit()

        resp_b = client.post(VERIFY_PATH, json={"token": token_b})
        assert resp_b.status_code == 200

        resp_a = client.post(VERIFY_PATH, json={"token": token_a})
        assert resp_a.status_code == 400
        assert resp_a.get_json()["error"] == "token_expired"


class TestCeleryAndFinalize:
    def test_cas_claim_ignores_superseded(self, db):
        session = _session(db)
        delivery, _ = _delivery(db, session)
        delivery.superseded_at = datetime.now(UTC)
        session.email_delivery_id = str(uuid.uuid4())  # pointeur ailleurs
        db.session.commit()
        assert cas_claim_sending(session, delivery.email_delivery_id) == "ignore"

    def test_resolve_token_refuses_superseded(self, db):
        session = _session(db)
        delivery, _ = _delivery(db, session)
        delivery.superseded_at = datetime.now(UTC)
        db.session.commit()
        assert resolve_activation_token_for_delivery(delivery.email_delivery_id) is None

    def test_finalize_late_a_does_not_rewrite_pointer(self, db):
        session = _session(db)
        a, _ = _delivery(db, session, status=EMAIL_DELIVERY_QUEUED)
        a.status = "sending"
        b, _ = _delivery(db, session, kind=EMAIL_DELIVERY_KIND_RESEND)
        a.superseded_at = datetime.now(UTC)
        a.superseded_by_delivery_id = b.email_delivery_id
        db.session.commit()
        pointer = session.email_delivery_id

        applied = finalize_after_provider_accepted(
            session, email_delivery_id=a.email_delivery_id, message_id="mid-a"
        )
        assert applied is True
        db.session.refresh(session)
        db.session.refresh(a)
        assert session.email_delivery_id == pointer == b.email_delivery_id
        assert a.provider_accepted_at is not None

    def test_mark_failed_cas_does_not_downgrade_sent(self, db):
        session = _session(db)
        delivery, _ = _delivery(db, session, status=EMAIL_DELIVERY_SENT)
        delivery.provider_accepted_at = datetime.now(UTC)
        db.session.commit()
        ok = mark_delivery_failed(
            session, "late error", email_delivery_id=delivery.email_delivery_id
        )
        assert ok is False
        db.session.refresh(delivery)
        assert delivery.status == EMAIL_DELIVERY_SENT

    def test_sync_mirror_never_changes_pointer(self, db):
        session = _session(db)
        _a, _ = _delivery(db, session)
        b, _ = _delivery(
            db, session, kind=EMAIL_DELIVERY_KIND_RESEND, make_current=False
        )
        pointer = session.email_delivery_id
        assert sync_current_delivery_mirror(session, b) is False
        assert session.email_delivery_id == pointer


class TestSnapshotNonMutative:
    def test_snapshot_does_not_fail_stale_sending(self, db):
        session = _session(db)
        delivery, _ = _delivery(db, session, status="sending")
        delivery.sending_started_at = datetime.now(UTC) - timedelta(hours=1)
        db.session.commit()
        ok, _reason = can_start_new_delivery_snapshot(session)
        assert ok is True
        db.session.refresh(delivery)
        assert delivery.status == "sending"


class TestLegacyBoot:
    def test_empty_window_disabled(self, monkeypatch):
        monkeypatch.delenv(
            "ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_FROM_UTC", raising=False
        )
        monkeypatch.delenv(
            "ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_UNTIL_UTC", raising=False
        )
        assert get_legacy_acceptance_window() is None
        assert is_legacy_acceptance_active() is False

    def test_single_var_refused(self, monkeypatch):
        monkeypatch.setenv(
            "ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_FROM_UTC",
            "2026-07-26T18:00:00+00:00",
        )
        monkeypatch.delenv(
            "ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_UNTIL_UTC", raising=False
        )
        with pytest.raises(ActivationLegacyConfigError):
            get_legacy_acceptance_window()

    def test_window_over_35_min_refused(self, monkeypatch):
        monkeypatch.setenv(
            "ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_FROM_UTC",
            "2026-07-26T18:00:00+00:00",
        )
        monkeypatch.setenv(
            "ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_UNTIL_UTC",
            "2026-07-26T19:00:00+00:00",
        )
        with pytest.raises(ActivationLegacyConfigError):
            validate_activation_legacy_for_boot(config_name="production")

    def test_naive_datetime_refused(self, monkeypatch):
        monkeypatch.setenv(
            "ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_FROM_UTC",
            "2026-07-26T18:00:00",
        )
        monkeypatch.setenv(
            "ACTIVATION_LEGACY_EMAIL_TOKEN_ACCEPT_UNTIL_UTC",
            "2026-07-26T18:20:00",
        )
        with pytest.raises(ActivationLegacyConfigError):
            get_legacy_acceptance_window()
