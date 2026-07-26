"""Tests unitaires : livraison email d'activation (Lot 1)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from models.activation_email_delivery import (
    EMAIL_DELIVERY_DELIVERED,
    EMAIL_DELIVERY_KIND_INITIAL,
    EMAIL_DELIVERY_KIND_RESEND,
    EMAIL_DELIVERY_QUEUED,
    EMAIL_DELIVERY_SENDING,
    EMAIL_DELIVERY_SENT,
    ActivationEmailDelivery,
)
from models.activation_session import ActivationSession
from services.notifications.activation_email_delivery import (
    cas_claim_sending,
    finalize_after_provider_accepted,
    sanitize_email_error,
    try_enqueue_activation_email,
)
from services.notifications.activation_token import (
    derive_activation_token,
    hash_activation_token,
)
from services.notifications.email_errors import (
    EmailPermanentError,
    EmailRetryableError,
)


def _make_session(db) -> ActivationSession:
    session = ActivationSession()
    session.activation_session_id = str(uuid.uuid4())
    session.user_id = 1
    session.sms_attempts = 0
    session.resend_count_email = 0
    session.resend_count_sms = 0
    db.session.add(session)
    db.session.commit()
    return session


def _make_delivery(
    db,
    session: ActivationSession,
    *,
    status: str = EMAIL_DELIVERY_QUEUED,
    kind: str = EMAIL_DELIVERY_KIND_INITIAL,
    delivery_id: str | None = None,
) -> ActivationEmailDelivery:
    did = delivery_id or str(uuid.uuid4())
    token = derive_activation_token(did, key_version=1)
    delivery = ActivationEmailDelivery(
        activation_session_pk=session.id,
        email_delivery_id=did,
        kind=kind,
        status=status,
        token_key_version=1,
        email_token_hash=hash_activation_token(token),
        token_expires_at=datetime.now(UTC) + timedelta(minutes=30),
    )
    db.session.add(delivery)
    session.email_delivery_id = did
    session.email_delivery_status = status
    session.email_delivery_kind = kind
    session.email_token_hash = delivery.email_token_hash
    db.session.commit()
    return delivery


class TestSanitizeEmailError:
    def test_strips_url_token_and_email(self):
        raw = (
            "fail https://www.lirie.ch/activate-account?token=abc.def.ghi "
            "user@example.com secret"
        )
        cleaned = sanitize_email_error(raw)
        assert "https://" not in cleaned
        assert "user@example.com" not in cleaned
        assert "[url]" in cleaned
        assert "[email]" in cleaned


class TestHmacToken:
    def test_same_delivery_id_same_token(self, monkeypatch):
        monkeypatch.setenv("ACTIVATION_TOKEN_KEY_V1", "test-activation-key-v1")
        did = str(uuid.uuid4())
        t1 = derive_activation_token(did, key_version=1)
        t2 = derive_activation_token(did, key_version=1)
        assert t1 == t2

    def test_new_delivery_id_new_token(self, monkeypatch):
        monkeypatch.setenv("ACTIVATION_TOKEN_KEY_V1", "test-activation-key-v1")
        t1 = derive_activation_token(str(uuid.uuid4()), key_version=1)
        t2 = derive_activation_token(str(uuid.uuid4()), key_version=1)
        assert t1 != t2


class TestTryEnqueueActivationEmail:
    def test_prod_config_missing_returns_502_flag(self, db):
        session = _make_session(db)

        with patch(
            "services.notifications.activation_email_delivery.is_email_provider_configured",
            return_value=(False, "Brevo not configured"),
        ):
            result = try_enqueue_activation_email(
                session,
                kind=EMAIL_DELIVERY_KIND_INITIAL,
                environment="production",
                is_testing=False,
            )

        assert result["require_502"] is True
        assert result["queued"] is False
        assert result["email_sent"] is None
        db.session.refresh(session)
        # F-03 : préflight provider — A inchangée, pas de supersession / failed
        assert session.email_delivery_status != "failed"
        assert session.email_delivery_id is None

    def test_enqueue_passes_session_and_delivery_id_only(self, db, monkeypatch):
        monkeypatch.setenv("ACTIVATION_TOKEN_KEY_V1", "test-activation-key-v1")
        session = _make_session(db)

        delay_mock = MagicMock()
        with (
            patch(
                "services.notifications.activation_email_delivery.is_email_provider_configured",
                return_value=(True, None),
            ),
            patch(
                "services.notifications.activation_email_delivery.enqueue_activation_email",
                delay_mock,
            ),
        ):
            result = try_enqueue_activation_email(
                session,
                kind=EMAIL_DELIVERY_KIND_INITIAL,
                environment="testing",
                is_testing=True,
            )

        assert result["queued"] is True
        assert result["email_sent"] is None
        assert result["require_502"] is False
        delay_mock.assert_called_once()
        kwargs = delay_mock.call_args.kwargs
        assert kwargs["activation_session_id"] == session.activation_session_id
        assert "email_delivery_id" in kwargs
        assert "verification_link" not in kwargs
        assert "token" not in kwargs
        db.session.refresh(session)
        assert session.email_delivery_status == EMAIL_DELIVERY_QUEUED
        assert session.last_email_sent_at is None
        # Même delivery_id → même jeton HMAC
        token = result["email_token"]
        assert token == derive_activation_token(
            kwargs["email_delivery_id"], key_version=1
        )

    def test_delay_failure_fails_delivery(self, db, monkeypatch):
        monkeypatch.setenv("ACTIVATION_TOKEN_KEY_V1", "test-activation-key-v1")
        session = _make_session(db)

        with (
            patch(
                "services.notifications.activation_email_delivery.is_email_provider_configured",
                return_value=(True, None),
            ),
            patch(
                "services.notifications.activation_email_delivery.enqueue_activation_email",
                side_effect=RuntimeError("broker down"),
            ),
        ):
            result = try_enqueue_activation_email(
                session,
                kind=EMAIL_DELIVERY_KIND_RESEND,
                environment="production",
                is_testing=False,
            )

        assert result["require_502"] is True
        db.session.refresh(session)
        assert session.email_delivery_status == "failed"


class TestCasClaimSending:
    def test_queued_to_sending_then_retry_same_delivery(self, db, monkeypatch):
        monkeypatch.setenv("ACTIVATION_TOKEN_KEY_V1", "test-activation-key-v1")
        session = _make_session(db)
        delivery = _make_delivery(db, session, status=EMAIL_DELIVERY_QUEUED)

        assert cas_claim_sending(session, delivery.email_delivery_id) == "proceed"
        db.session.refresh(session)
        assert session.email_delivery_status == EMAIL_DELIVERY_SENDING

        assert cas_claim_sending(session, delivery.email_delivery_id) == "proceed"
        assert cas_claim_sending(session, str(uuid.uuid4())) == "ignore"


class TestFinalizeAfterProviderAccepted:
    def test_resend_increments_count_only_once(self, db, monkeypatch):
        monkeypatch.setenv("ACTIVATION_TOKEN_KEY_V1", "test-activation-key-v1")
        session = _make_session(db)
        session.resend_count_email = 2
        session.last_email_sent_at = datetime.now(UTC) - timedelta(minutes=5)
        delivery = _make_delivery(
            db,
            session,
            status=EMAIL_DELIVERY_SENDING,
            kind=EMAIL_DELIVERY_KIND_RESEND,
        )
        db.session.commit()

        applied = finalize_after_provider_accepted(
            session, email_delivery_id=delivery.email_delivery_id, message_id="msg-1"
        )
        db.session.commit()
        assert applied is True
        db.session.refresh(session)
        assert session.email_delivery_status == EMAIL_DELIVERY_SENT
        assert session.resend_count_email == 3
        assert session.email_provider_message_id == "msg-1"

        # Second appel → no-op compteur
        applied2 = finalize_after_provider_accepted(
            session, email_delivery_id=delivery.email_delivery_id, message_id="msg-2"
        )
        db.session.commit()
        assert applied2 is False
        db.session.refresh(session)
        assert session.resend_count_email == 3

    def test_webhook_delivered_before_finalize_keeps_delivered(self, db, monkeypatch):
        monkeypatch.setenv("ACTIVATION_TOKEN_KEY_V1", "test-activation-key-v1")
        session = _make_session(db)
        session.resend_count_email = 0
        delivery = _make_delivery(
            db,
            session,
            status=EMAIL_DELIVERY_SENDING,
            kind=EMAIL_DELIVERY_KIND_RESEND,
        )
        # Webhook intercalé
        delivery.status = EMAIL_DELIVERY_DELIVERED
        session.email_delivery_status = EMAIL_DELIVERY_DELIVERED
        db.session.commit()

        applied = finalize_after_provider_accepted(
            session, email_delivery_id=delivery.email_delivery_id, message_id="mid"
        )
        db.session.commit()
        assert applied is True
        db.session.refresh(delivery)
        db.session.refresh(session)
        assert delivery.status == EMAIL_DELIVERY_DELIVERED
        assert delivery.provider_accepted_at is not None
        assert session.resend_count_email == 1


class TestBrevoTransactional:
    @patch("services.email.brevo_provider.requests.post")
    def test_send_transactional_201(self, mock_post):
        from services.email.brevo_provider import BrevoEmailProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 201
        mock_resp.text = '{"messageId":"m1"}'
        mock_resp.json.return_value = {"messageId": "m1"}
        mock_post.return_value = mock_resp

        provider = BrevoEmailProvider(api_key="k")
        result = provider.send_transactional(
            to_email="a@b.ch",
            subject="Activation",
            text_content="Bienvenue sur LIRIE",
            from_email="noreply@lirie.ch",
            from_name="LIRIE",
            reply_to="support@lirie.ch",
            headers={"X-Mailin-custom": "delivery-uuid"},
        )
        assert result.success is True
        assert result.message_id == "m1"
        payload = mock_post.call_args.kwargs["json"]
        assert payload["sender"]["email"] == "noreply@lirie.ch"
        assert payload["headers"]["X-Mailin-custom"] == "delivery-uuid"

    @patch("services.email.brevo_provider.requests.post")
    def test_send_transactional_429_retryable(self, mock_post):
        from services.email.brevo_provider import BrevoEmailProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.text = "rate"
        mock_post.return_value = mock_resp

        provider = BrevoEmailProvider(api_key="k")
        result = provider.send_transactional(
            to_email="a@b.ch",
            subject="x",
            text_content="y",
            from_email="noreply@lirie.ch",
            from_name="LIRIE",
        )
        assert result.success is False
        assert result.retryable is True


class TestSendEmailNotificationRaises:
    def test_raise_on_error_maps_retryable(self):
        with (
            patch(
                "services.notifications.email.EMAIL_ENABLED",
                True,
            ),
            patch(
                "services.notifications.email.EMAIL_PROVIDER",
                "brevo",
            ),
            patch(
                "services.notifications.email.BREVO_API_KEY",
                "key",
            ),
            patch("services.email.brevo_provider.BrevoEmailProvider") as provider_cls,
        ):
            instance = provider_cls.return_value
            instance.send_transactional.return_value = MagicMock(
                success=False,
                error="timeout",
                retryable=True,
                status_code=None,
                message_id=None,
            )
            from services.notifications.email import send_email_notification

            with pytest.raises(EmailRetryableError):
                send_email_notification(
                    "a@b.ch",
                    "s",
                    "body",
                    raise_on_error=True,
                )

    def test_raise_on_error_maps_permanent(self):
        with (
            patch(
                "services.notifications.email.EMAIL_ENABLED",
                True,
            ),
            patch(
                "services.notifications.email.EMAIL_PROVIDER",
                "brevo",
            ),
            patch(
                "services.notifications.email.BREVO_API_KEY",
                "key",
            ),
            patch("services.email.brevo_provider.BrevoEmailProvider") as provider_cls,
        ):
            instance = provider_cls.return_value
            instance.send_transactional.return_value = MagicMock(
                success=False,
                error="bad",
                retryable=False,
                status_code=400,
                message_id=None,
            )
            from services.notifications.email import send_email_notification

            with pytest.raises(EmailPermanentError):
                send_email_notification(
                    "a@b.ch",
                    "s",
                    "body",
                    raise_on_error=True,
                )
