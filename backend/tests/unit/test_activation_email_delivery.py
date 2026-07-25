"""Tests unitaires : livraison email d'activation (P0/P1)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from models.activation_session import (
    EMAIL_DELIVERY_FAILED,
    EMAIL_DELIVERY_KIND_INITIAL,
    EMAIL_DELIVERY_KIND_RESEND,
    EMAIL_DELIVERY_QUEUED,
    EMAIL_DELIVERY_SENDING,
    EMAIL_DELIVERY_SENT,
    ActivationSession,
)
from services.notifications.activation_email_delivery import (
    cas_claim_sending,
    mark_delivery_sent,
    sanitize_email_error,
    try_enqueue_activation_email,
)
from services.notifications.email_errors import (
    EmailPermanentError,
    EmailRetryableError,
)


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


class TestTryEnqueueActivationEmail:
    def test_prod_config_missing_returns_502_flag(self, db):
        session = ActivationSession()
        session.activation_session_id = str(uuid.uuid4())
        session.user_id = 1
        session.sms_attempts = 0
        session.resend_count_email = 0
        session.resend_count_sms = 0
        db.session.add(session)
        db.session.commit()

        with patch(
            "services.notifications.activation_email_delivery.is_email_provider_configured",
            return_value=(False, "Brevo not configured"),
        ):
            result = try_enqueue_activation_email(
                session,
                kind=EMAIL_DELIVERY_KIND_INITIAL,
                email_token="plain-token-value",
                email_token_hash="abc123",
                environment="production",
                is_testing=False,
            )

        assert result["require_502"] is True
        assert result["queued"] is False
        assert result["email_sent"] is None
        db.session.refresh(session)
        assert session.email_delivery_status == EMAIL_DELIVERY_FAILED

    def test_enqueue_passes_session_and_delivery_id_only(self, db):
        session = ActivationSession()
        session.activation_session_id = str(uuid.uuid4())
        session.user_id = 1
        session.sms_attempts = 0
        session.resend_count_email = 0
        session.resend_count_sms = 0
        db.session.add(session)
        db.session.commit()

        delay_mock = MagicMock()
        with (
            patch(
                "services.notifications.activation_email_delivery.is_email_provider_configured",
                return_value=(True, None),
            ),
            patch(
                "services.notifications.activation_email_delivery.store_activation_email_token"
            ),
            patch(
                "services.notifications.activation_email_delivery.enqueue_activation_email",
                delay_mock,
            ),
        ):
            result = try_enqueue_activation_email(
                session,
                kind=EMAIL_DELIVERY_KIND_INITIAL,
                email_token="plain-token-value",
                email_token_hash="hash-xyz",
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

    def test_delay_failure_purges_and_fails(self, db):
        session = ActivationSession()
        session.activation_session_id = str(uuid.uuid4())
        session.user_id = 1
        session.sms_attempts = 0
        session.resend_count_email = 0
        session.resend_count_sms = 0
        db.session.add(session)
        db.session.commit()

        with (
            patch(
                "services.notifications.activation_email_delivery.is_email_provider_configured",
                return_value=(True, None),
            ),
            patch(
                "services.notifications.activation_email_delivery.store_activation_email_token"
            ),
            patch(
                "services.notifications.activation_email_delivery.purge_activation_email_token"
            ) as purge,
            patch(
                "services.notifications.activation_email_delivery.enqueue_activation_email",
                side_effect=RuntimeError("broker down"),
            ),
        ):
            result = try_enqueue_activation_email(
                session,
                kind=EMAIL_DELIVERY_KIND_RESEND,
                email_token="plain-token",
                email_token_hash="h",
                environment="production",
                is_testing=False,
            )

        assert result["require_502"] is True
        purge.assert_called_once()
        db.session.refresh(session)
        assert session.email_delivery_status == EMAIL_DELIVERY_FAILED


class TestCasClaimSending:
    def test_queued_to_sending_then_retry_same_delivery(self, db):
        delivery_id = str(uuid.uuid4())
        session = ActivationSession()
        session.activation_session_id = str(uuid.uuid4())
        session.user_id = 1
        session.sms_attempts = 0
        session.resend_count_email = 0
        session.resend_count_sms = 0
        session.email_delivery_id = delivery_id
        session.email_delivery_status = EMAIL_DELIVERY_QUEUED
        db.session.add(session)
        db.session.commit()

        assert cas_claim_sending(session, delivery_id) == "proceed"
        db.session.refresh(session)
        assert session.email_delivery_status == EMAIL_DELIVERY_SENDING

        # Retry même delivery_id
        assert cas_claim_sending(session, delivery_id) == "proceed"

        # Ancien job
        assert cas_claim_sending(session, str(uuid.uuid4())) == "ignore"


class TestMarkDeliverySent:
    def test_resend_increments_count_only_on_success(self, db):
        delivery_id = str(uuid.uuid4())
        session = ActivationSession()
        session.activation_session_id = str(uuid.uuid4())
        session.user_id = 1
        session.sms_attempts = 0
        session.resend_count_email = 2
        session.resend_count_sms = 0
        session.email_delivery_id = delivery_id
        session.email_delivery_kind = EMAIL_DELIVERY_KIND_RESEND
        session.email_delivery_status = EMAIL_DELIVERY_SENDING
        session.last_email_sent_at = datetime.now(UTC) - timedelta(minutes=5)
        db.session.add(session)
        db.session.commit()

        with patch(
            "services.notifications.activation_email_delivery.purge_activation_email_token"
        ):
            mark_delivery_sent(
                session, email_delivery_id=delivery_id, message_id="msg-1"
            )
        db.session.commit()
        db.session.refresh(session)
        assert session.email_delivery_status == EMAIL_DELIVERY_SENT
        assert session.resend_count_email == 3
        assert session.email_provider_message_id == "msg-1"
        assert session.last_email_sent_at is not None


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
        )
        assert result.success is True
        assert result.message_id == "m1"
        payload = mock_post.call_args.kwargs["json"]
        assert payload["sender"]["email"] == "noreply@lirie.ch"
        assert payload["sender"]["name"] == "LIRIE"
        assert payload["replyTo"]["email"] == "support@lirie.ch"
        assert mock_post.call_args.kwargs["timeout"] == provider.http_timeout

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

    @patch("services.email.brevo_provider.requests.post")
    def test_send_transactional_400_not_retryable(self, mock_post):
        from services.email.brevo_provider import BrevoEmailProvider

        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = "bad"
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
        assert result.retryable is False


class TestSendEmailNotificationRaises:
    def test_raise_on_error_maps_retryable(self):
        with patch(
            "services.notifications.email.EMAIL_ENABLED",
            True,
        ), patch(
            "services.notifications.email.EMAIL_PROVIDER",
            "brevo",
        ), patch(
            "services.notifications.email.BREVO_API_KEY",
            "key",
        ), patch(
            "services.email.brevo_provider.BrevoEmailProvider"
        ) as provider_cls:
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
        with patch(
            "services.notifications.email.EMAIL_ENABLED",
            True,
        ), patch(
            "services.notifications.email.EMAIL_PROVIDER",
            "brevo",
        ), patch(
            "services.notifications.email.BREVO_API_KEY",
            "key",
        ), patch(
            "services.email.brevo_provider.BrevoEmailProvider"
        ) as provider_cls:
            instance = provider_cls.return_value
            instance.send_transactional.return_value = MagicMock(
                success=False,
                error="bad request",
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


class TestActivationEmailTask:
    def test_stale_delivery_ignored(self, app, db):
        from tasks.notification_tasks import send_activation_email_task

        delivery_current = str(uuid.uuid4())
        session = ActivationSession()
        session.activation_session_id = str(uuid.uuid4())
        session.user_id = 1
        session.sms_attempts = 0
        session.resend_count_email = 0
        session.resend_count_sms = 0
        session.email_delivery_id = delivery_current
        session.email_delivery_status = EMAIL_DELIVERY_SENDING
        db.session.add(session)
        db.session.commit()

        with app.app_context():
            result = send_activation_email_task.run(
                activation_session_id=session.activation_session_id,
                email_delivery_id=str(uuid.uuid4()),
            )
        assert result.get("ignored") is True

    def test_success_marks_sent_same_token_hash(self, app, db):
        from models import User
        from tasks.notification_tasks import send_activation_email_task

        user = User(
            username=f"act_{uuid.uuid4().hex[:8]}",
            email=f"act_{uuid.uuid4().hex[:8]}@example.com",
        )
        user.set_password("TestPass123!")
        db.session.add(user)
        db.session.flush()

        delivery_id = str(uuid.uuid4())
        token_hash = "stable-hash-abc"
        session = ActivationSession()
        session.activation_session_id = str(uuid.uuid4())
        session.user_id = user.id
        session.sms_attempts = 0
        session.resend_count_email = 0
        session.resend_count_sms = 0
        session.email_delivery_id = delivery_id
        session.email_delivery_kind = EMAIL_DELIVERY_KIND_INITIAL
        session.email_delivery_status = EMAIL_DELIVERY_QUEUED
        session.email_token_hash = token_hash
        db.session.add(session)
        db.session.commit()

        with (
            app.app_context(),
            patch(
                "services.notifications.activation_email_delivery.get_activation_email_token",
                return_value="stable-plain-token",
            ),
            patch(
                "services.notifications.email.send_email_notification",
                return_value={"ok": True, "message_id": "mid-9"},
            ),
            patch(
                "services.notifications.activation_email_delivery.purge_activation_email_token"
            ),
        ):
            result = send_activation_email_task.run(
                activation_session_id=session.activation_session_id,
                email_delivery_id=delivery_id,
            )

        assert result.get("ok") is True
        db.session.refresh(session)
        assert session.email_delivery_status == EMAIL_DELIVERY_SENT
        assert session.email_token_hash == token_hash
        assert session.last_email_sent_at is not None
        assert session.email_provider_message_id == "mid-9"
