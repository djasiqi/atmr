"""Tests unitaires pour les alertes IP whitelist.

Valide le fonctionnement du système d'alertes pour les tentatives d'accès
non autorisées via IP whitelist :
- Enregistrement dans l'audit log
- Envoi d'alertes Sentry
- Rate limiting pour éviter le spam
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from security.ip_whitelist_alerts import (
    ALERT_RATE_LIMIT_MINUTES,
    REDIS_ALERT_KEY_PREFIX,
    send_ip_whitelist_alert,
    should_alert_for_ip,
)


@pytest.fixture
def mock_redis_client():
    """Mock Redis client."""
    with patch("security.ip_whitelist_alerts.redis_client") as mock_redis:
        mock_redis.exists = MagicMock(return_value=0)
        mock_redis.setex = MagicMock()
        yield mock_redis


@pytest.fixture
def mock_audit_logger():
    """Mock AuditLogger."""
    with patch("security.ip_whitelist_alerts.AuditLogger") as mock_logger:
        mock_log_instance = MagicMock()
        mock_logger.log_security_event.return_value = mock_log_instance
        yield mock_logger


@pytest.fixture
def mock_request():
    """Mock Flask request."""
    with patch("security.ip_whitelist_alerts.request") as mock_req:
        mock_req.headers = {
            "User-Agent": "Mozilla/5.0",
            "X-Forwarded-For": "192.168.1.100",
        }
        yield mock_req


class TestShouldAlertForIP:
    """Tests pour la fonction should_alert_for_ip (rate limiting)."""

    def test_should_alert_for_ip_redis_available_first_time(self, mock_redis_client):
        """Test rate limiting avec Redis: première alerte autorisée."""
        mock_redis_client.exists.return_value = 0  # Pas d'alerte récente

        result = should_alert_for_ip("192.168.1.100")

        assert result is True
        mock_redis_client.exists.assert_called_once_with(
            f"{REDIS_ALERT_KEY_PREFIX}192.168.1.100"
        )
        mock_redis_client.setex.assert_called_once()
        args, _ = mock_redis_client.setex.call_args
        assert args[0] == f"{REDIS_ALERT_KEY_PREFIX}192.168.1.100"
        assert args[1] == ALERT_RATE_LIMIT_MINUTES * 60
        assert args[2] == "1"

    def test_should_alert_for_ip_redis_available_rate_limited(self, mock_redis_client):
        """Test rate limiting avec Redis: alerte récente, bloquée."""
        mock_redis_client.exists.return_value = 1  # Alerte récente existe

        result = should_alert_for_ip("192.168.1.100")

        assert result is False
        mock_redis_client.exists.assert_called_once_with(
            f"{REDIS_ALERT_KEY_PREFIX}192.168.1.100"
        )
        mock_redis_client.setex.assert_not_called()

    def test_should_alert_for_ip_redis_unavailable_fallback_memory(self):
        """Test rate limiting sans Redis: fallback vers cache mémoire."""
        with patch("security.ip_whitelist_alerts.redis_client", None):
            # Première alerte: autorisée
            result1 = should_alert_for_ip("192.168.1.200")
            assert result1 is True

            # Deuxième alerte immédiate: bloquée (rate limit)
            result2 = should_alert_for_ip("192.168.1.200")
            assert result2 is False

            # Alerte après expiration: autorisée
            with patch("security.ip_whitelist_alerts.datetime") as mock_datetime:
                mock_datetime.now.return_value = datetime.now(UTC) + timedelta(
                    minutes=ALERT_RATE_LIMIT_MINUTES + 1
                )
                # Réinitialiser le cache mémoire
                if hasattr(should_alert_for_ip, "_memory_cache"):
                    should_alert_for_ip._memory_cache = {}  # type: ignore[attr-defined]

                result3 = should_alert_for_ip("192.168.1.200")
                assert result3 is True

    def test_should_alert_for_ip_redis_error_fail_open(self, mock_redis_client):
        """Test rate limiting: erreur Redis -> fail-open (autorise l'alerte)."""
        mock_redis_client.exists.side_effect = Exception("Redis connection error")

        result = should_alert_for_ip("192.168.1.100")

        # En cas d'erreur, on autorise l'alerte (fail-open)
        assert result is True


class TestSendIPWhitelistAlert:
    """Tests pour la fonction send_ip_whitelist_alert."""

    @patch("security.ip_whitelist_alerts.sentry_sdk.capture_message")
    @patch("security.ip_whitelist_alerts.should_alert_for_ip")
    def test_send_alert_audit_log_called(
        self, mock_should_alert, mock_sentry_capture, mock_audit_logger, mock_request
    ):
        """Test que AuditLogger est appelé lors d'une alerte."""
        mock_should_alert.return_value = True

        send_ip_whitelist_alert(
            client_ip="192.168.1.100",
            endpoint="/api/admin/stats",
            method="GET",
            user_agent="Mozilla/5.0",
        )

        # Vérifier que AuditLogger.log_security_event a été appelé
        mock_audit_logger.log_security_event.assert_called_once()
        call_args = mock_audit_logger.log_security_event.call_args

        assert call_args.kwargs["event_type"] == "ip_whitelist_denied"
        assert call_args.kwargs["severity"] == "high"
        assert call_args.kwargs["ip_address"] == "192.168.1.100"
        assert call_args.kwargs["user_agent"] == "Mozilla/5.0"

        details = call_args.kwargs["details"]
        assert details["endpoint"] == "/api/admin/stats"
        assert details["method"] == "GET"
        assert details["user_agent"] == "Mozilla/5.0"

    @patch("security.ip_whitelist_alerts.sentry_sdk.capture_message")
    @patch("security.ip_whitelist_alerts.should_alert_for_ip")
    def test_send_alert_sentry_called_when_allowed(
        self, mock_should_alert, mock_sentry_capture, mock_audit_logger, mock_request
    ):
        """Test que Sentry est appelé quand le rate limiting l'autorise."""
        mock_should_alert.return_value = True

        send_ip_whitelist_alert(
            client_ip="192.168.1.100",
            endpoint="/api/admin/stats",
            method="GET",
            user_agent="Mozilla/5.0",
            user_id=123,
        )

        # Vérifier que Sentry a été appelé
        mock_sentry_capture.assert_called_once()
        call_args = mock_sentry_capture.call_args

        assert "Tentative d'accès non autorisée via IP whitelist" in call_args[0][0]
        assert call_args[1]["level"] == "warning"

        tags = call_args[1]["tags"]
        assert tags["security_event"] == "ip_whitelist_denied"
        assert tags["ip_address"] == "192.168.1.100"
        assert tags["endpoint"] == "/api/admin/stats"
        assert tags["method"] == "GET"

        context = call_args[1]["contexts"]["request"]
        assert context["ip_address"] == "192.168.1.100"
        assert context["endpoint"] == "/api/admin/stats"
        assert context["user_id"] == 123

    @patch("security.ip_whitelist_alerts.sentry_sdk.capture_message")
    @patch("security.ip_whitelist_alerts.should_alert_for_ip")
    def test_send_alert_sentry_not_called_when_rate_limited(
        self, mock_should_alert, mock_sentry_capture, mock_audit_logger, mock_request
    ):
        """Test que Sentry n'est pas appelé quand le rate limiting bloque."""
        mock_should_alert.return_value = False  # Rate limited

        send_ip_whitelist_alert(
            client_ip="192.168.1.100",
            endpoint="/api/admin/stats",
            method="GET",
        )

        # Audit log doit toujours être appelé
        mock_audit_logger.log_security_event.assert_called_once()

        # Mais Sentry ne doit pas être appelé
        mock_sentry_capture.assert_not_called()

    @patch("security.ip_whitelist_alerts.sentry_sdk.capture_message")
    @patch("security.ip_whitelist_alerts.should_alert_for_ip")
    def test_send_alert_with_headers(
        self, mock_should_alert, mock_sentry_capture, mock_audit_logger, mock_request
    ):
        """Test que les headers de sécurité sont collectés."""
        mock_should_alert.return_value = True
        mock_request.headers = {
            "User-Agent": "Mozilla/5.0",
            "X-Forwarded-For": "192.168.1.100, 10.0.0.1",
            "X-Real-IP": "192.168.1.100",
            "Origin": "https://example.com",
            "Referer": "https://example.com/page",
        }

        send_ip_whitelist_alert(
            client_ip="192.168.1.100",
            endpoint="/api/admin/stats",
            method="GET",
        )

        call_args = mock_audit_logger.log_security_event.call_args
        details = call_args.kwargs["details"]

        assert "headers" in details
        assert "X-Forwarded-For" in details["headers"]
        assert "X-Real-IP" in details["headers"]
        assert "Origin" in details["headers"]
        assert "Referer" in details["headers"]

    @patch("security.ip_whitelist_alerts.sentry_sdk.capture_message")
    @patch("security.ip_whitelist_alerts.should_alert_for_ip")
    def test_send_alert_without_user_id(
        self, mock_should_alert, mock_sentry_capture, mock_audit_logger, mock_request
    ):
        """Test alerte sans user_id (utilisateur non authentifié)."""
        mock_should_alert.return_value = True

        send_ip_whitelist_alert(
            client_ip="192.168.1.100",
            endpoint="/api/admin/stats",
            method="GET",
        )

        call_args = mock_audit_logger.log_security_event.call_args
        assert call_args.kwargs.get("user_id") is None

        sentry_call_args = mock_sentry_capture.call_args
        context = sentry_call_args[1]["contexts"]["request"]
        assert "user_id" not in context

    @patch("security.ip_whitelist_alerts.sentry_sdk.capture_message")
    @patch("security.ip_whitelist_alerts.should_alert_for_ip")
    def test_send_alert_exception_handling(
        self, mock_should_alert, mock_sentry_capture, mock_audit_logger, mock_request
    ):
        """Test que les exceptions ne font pas échouer la fonction."""
        mock_should_alert.return_value = True
        mock_audit_logger.log_security_event.side_effect = Exception("DB error")

        # Ne doit pas lever d'exception
        send_ip_whitelist_alert(
            client_ip="192.168.1.100",
            endpoint="/api/admin/stats",
            method="GET",
        )

        # L'exception doit être loggée mais pas propagée
        mock_audit_logger.log_security_event.assert_called_once()


class TestIPWhitelistAlertIntegration:
    """Tests d'intégration pour les alertes IP whitelist."""

    @patch("security.ip_whitelist_alerts.sentry_sdk.capture_message")
    @patch("security.ip_whitelist_alerts.should_alert_for_ip")
    def test_full_alert_flow(
        self, mock_should_alert, mock_sentry_capture, mock_audit_logger, mock_request
    ):
        """Test du flux complet d'alerte."""
        mock_should_alert.return_value = True

        send_ip_whitelist_alert(
            client_ip="192.168.1.100",
            endpoint="/api/admin/stats",
            method="POST",
            user_agent="curl/7.68.0",
            user_id=456,
        )

        # Vérifier que tout a été appelé dans le bon ordre
        assert mock_audit_logger.log_security_event.called
        assert mock_sentry_capture.called

        # Vérifier les détails de l'alerte Sentry
        sentry_call = mock_sentry_capture.call_args
        assert "192.168.1.100" in sentry_call[0][0]
        assert sentry_call[1]["tags"]["endpoint"] == "/api/admin/stats"
        assert sentry_call[1]["tags"]["method"] == "POST"
