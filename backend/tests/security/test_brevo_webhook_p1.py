"""Lot 1-B : webhooks Brevo — Bearer, ON CONFLICT, txn atomique."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest

from models.activation_email_delivery import (
    EMAIL_DELIVERY_DELIVERED,
    EMAIL_DELIVERY_KIND_INITIAL,
    EMAIL_DELIVERY_SENT,
    EMAIL_DELIVERY_SPAM,
    ActivationEmailDelivery,
    BrevoWebhookEvent,
)
from models.activation_session import ActivationSession
from services.notifications.activation_token import (
    derive_activation_token,
    hash_activation_token,
)
from services.notifications.brevo_webhook import (
    compute_idempotency_key,
    process_brevo_webhook_event,
    verify_brevo_bearer,
)


@pytest.fixture
def brevo_secret(monkeypatch):
    monkeypatch.setenv("BREVO_WEBHOOK_SECRET", "test-secret-lot1")
    monkeypatch.setenv("ACTIVATION_TOKEN_KEY_V1", "test-activation-key-v1")
    return "test-secret-lot1"


def _session_with_delivery(db_session, *, status=EMAIL_DELIVERY_SENT, mid=None):
    session = ActivationSession()
    session.activation_session_id = str(uuid.uuid4())
    session.user_id = 1
    session.sms_attempts = 0
    session.resend_count_email = 0
    session.resend_count_sms = 0
    db_session.add(session)
    db_session.flush()
    did = str(uuid.uuid4())
    token = derive_activation_token(did, key_version=1)
    delivery = ActivationEmailDelivery(
        activation_session_pk=session.id,
        email_delivery_id=did,
        kind=EMAIL_DELIVERY_KIND_INITIAL,
        status=status,
        token_key_version=1,
        email_token_hash=hash_activation_token(token),
        token_expires_at=datetime.now(UTC) + timedelta(minutes=30),
        provider_message_id=mid,
    )
    db_session.add(delivery)
    session.email_delivery_id = did
    session.email_delivery_status = status
    db_session.commit()
    return session, delivery


def _count_events(ikey: str) -> int:
    return BrevoWebhookEvent.query.filter_by(idempotency_key=ikey).count()


class TestBrevoBearer:
    def test_invalid_bearer_401(self, client, brevo_secret):
        resp = client.post(
            "/api/v1/webhooks/brevo",
            json={"event": "delivered", "message-id": "m1", "email": "a@b.ch"},
            headers={"Authorization": "Bearer wrong"},
        )
        assert resp.status_code == 401

    def test_valid_bearer_ok(self, client, db, brevo_secret):
        _session, delivery = _session_with_delivery(
            db.session, mid="<msg@brevo>"
        )
        resp = client.post(
            "/api/v1/webhooks/brevo",
            json={
                "event": "delivered",
                "message-id": "<msg@brevo>",
                "email": "a@b.ch",
                "ts_event": "1",
                "X-Mailin-custom": delivery.email_delivery_id,
            },
            headers={"Authorization": f"Bearer {brevo_secret}"},
        )
        assert resp.status_code == 200
        assert resp.get_json()["status"] in {"ok", "noop", "ignored"}


class TestBrevoIdempotence:
    def test_on_conflict_noop_without_abort(self, db, brevo_secret):
        mid = f"m-dup-{uuid.uuid4()}"
        _session, delivery = _session_with_delivery(db.session, mid=mid)
        payload = {
            "event": "delivered",
            "message-id": mid,
            "email": "a@b.ch",
            "ts_event": "99",
            "X-Mailin-custom": delivery.email_delivery_id,
        }
        ikey = compute_idempotency_key(
            message_id=mid, event="delivered", ts_event="99", email="a@b.ch"
        )
        r1 = process_brevo_webhook_event(payload)
        assert r1["status"] == "ok"
        assert _count_events(ikey) == 1
        r2 = process_brevo_webhook_event(payload)
        assert r2["status"] == "noop"
        assert _count_events(ikey) == 1

    def test_delivered_then_spam_two_keys(self, db, brevo_secret):
        mid = f"m-spam-{uuid.uuid4()}"
        _session, delivery = _session_with_delivery(db.session, mid=mid)
        base = {
            "message-id": mid,
            "email": "a@b.ch",
            "ts_event": "1",
            "X-Mailin-custom": delivery.email_delivery_id,
        }
        assert process_brevo_webhook_event({**base, "event": "delivered"})["status"] == "ok"
        db.session.refresh(delivery)
        assert delivery.status == EMAIL_DELIVERY_DELIVERED
        assert process_brevo_webhook_event({**base, "event": "spam", "ts_event": "2"})[
            "status"
        ] == "ok"
        db.session.refresh(delivery)
        assert delivery.status == EMAIL_DELIVERY_SPAM
        k1 = compute_idempotency_key(
            message_id=mid, event="delivered", ts_event="1", email="a@b.ch"
        )
        k2 = compute_idempotency_key(
            message_id=mid, event="spam", ts_event="2", email="a@b.ch"
        )
        assert _count_events(k1) == 1
        assert _count_events(k2) == 1

    def test_custom_header_without_message_id_in_db(self, db, brevo_secret):
        _session, delivery = _session_with_delivery(db.session, mid=None)
        mid = f"new-mid-{uuid.uuid4()}"
        r = process_brevo_webhook_event(
            {
                "event": "delivered",
                "message-id": mid,
                "email": "a@b.ch",
                "ts_event": "1",
                "X-Mailin-custom": delivery.email_delivery_id,
            }
        )
        assert r["status"] == "ok"
        db.session.refresh(delivery)
        assert delivery.status == EMAIL_DELIVERY_DELIVERED
        assert delivery.provider_message_id == mid

    def test_old_delivery_bounce_leaves_current(self, db, brevo_secret):
        cur_mid = f"cur-{uuid.uuid4()}"
        old_mid = f"old-mid-{uuid.uuid4()}"
        session, current = _session_with_delivery(db.session, mid=cur_mid)
        old_id = str(uuid.uuid4())
        old = ActivationEmailDelivery(
            activation_session_pk=session.id,
            email_delivery_id=old_id,
            kind=EMAIL_DELIVERY_KIND_INITIAL,
            status=EMAIL_DELIVERY_SENT,
            token_key_version=1,
            email_token_hash="x" * 64,
            provider_message_id=old_mid,
        )
        db.session.add(old)
        db.session.commit()

        process_brevo_webhook_event(
            {
                "event": "hard_bounce",
                "message-id": old_mid,
                "email": "a@b.ch",
                "ts_event": "1",
                "X-Mailin-custom": old_id,
            }
        )
        db.session.refresh(current)
        db.session.refresh(old)
        assert current.status == EMAIL_DELIVERY_SENT
        assert old.status == "hard_bounced"
        assert session.email_delivery_id == current.email_delivery_id

    def test_mid_txn_error_rolls_back_idempotency(self, db, brevo_secret):
        mid = f"m-rb-{uuid.uuid4()}"
        _session, delivery = _session_with_delivery(db.session, mid=mid)
        payload = {
            "event": "delivered",
            "message-id": mid,
            "email": "a@b.ch",
            "ts_event": "1",
            "X-Mailin-custom": delivery.email_delivery_id,
        }
        ikey = compute_idempotency_key(
            message_id=mid, event="delivered", ts_event="1", email="a@b.ch"
        )
        with (
            patch(
                "services.notifications.brevo_webhook.apply_delivery_transition",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(RuntimeError),
        ):
            process_brevo_webhook_event(payload)
        assert _count_events(ikey) == 0
        r = process_brevo_webhook_event(payload)
        assert r["status"] == "ok"
        assert _count_events(ikey) == 1


class TestIdempotencyKey:
    def test_sha256_stable(self):
        k1 = compute_idempotency_key(
            message_id="m", event="delivered", ts_event="1", email="A@B.ch"
        )
        k2 = compute_idempotency_key(
            message_id="m", event="delivered", ts_event="1", email="a@b.ch"
        )
        assert k1 == k2
        assert len(k1) == 64


class TestVerifyBearer:
    def test_compare_digest(self, brevo_secret):
        assert verify_brevo_bearer(f"Bearer {brevo_secret}") is True
        assert verify_brevo_bearer("Bearer other") is False
        assert verify_brevo_bearer(None) is False
