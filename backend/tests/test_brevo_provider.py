"""
Tests unitaires pour le provider Brevo.

Ces tests utilisent des mocks pour ne pas nécessiter une vraie clé API.
"""

from unittest.mock import Mock, patch

import pytest

from services.email.brevo_provider import (
    BrevoEmailProvider,
    DomainVerificationResult,
    EmailResult,
)


class TestBrevoEmailProvider:
    """Tests du provider Brevo."""

    def test_init_with_api_key(self):
        """Test initialisation avec clé API fournie."""
        provider = BrevoEmailProvider(api_key="test_key")
        assert provider.api_key == "test_key"
        assert provider.base_url == "https://api.brevo.com/v3"

    def test_init_without_api_key_raises_error(self):
        """Test initialisation sans clé API lève une erreur."""
        with (
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(ValueError, match="Brevo API key manquante"),
        ):
            BrevoEmailProvider()

    def test_init_with_env_var(self):
        """Test initialisation avec variable d'environnement."""
        with patch.dict("os.environ", {"BREVO_API_KEY": "env_key"}):
            provider = BrevoEmailProvider()
            assert provider.api_key == "env_key"

    @patch("services.email.brevo_provider.requests.post")
    def test_send_invoice_email_success(self, mock_post):
        """Test envoi email réussi."""
        # Mock réponse Brevo
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"messageId": "msg-123"}
        mock_post.return_value = mock_response

        provider = BrevoEmailProvider(api_key="test_key")
        result = provider.send_invoice_email(
            from_email="noreply@test.ch",
            from_name="Test",
            to_email="client@example.com",
            to_name="Client",
            subject="Test",
            html_content="<p>Test</p>",
        )

        assert result.success is True
        assert result.message_id == "msg-123"
        assert result.error is None

        # Vérifier l'appel API
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://api.brevo.com/v3/smtp/email"
        payload = call_args[1]["json"]
        assert payload["sender"]["email"] == "noreply@test.ch"
        assert payload["to"][0]["email"] == "client@example.com"

    @patch("services.email.brevo_provider.requests.post")
    def test_send_invoice_email_with_attachment(self, mock_post):
        """Test envoi email avec pièce jointe."""
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"messageId": "msg-456"}
        mock_post.return_value = mock_response

        provider = BrevoEmailProvider(api_key="test_key")
        pdf_content = b"PDF content"

        result = provider.send_invoice_email(
            from_email="noreply@test.ch",
            from_name="Test",
            to_email="client@example.com",
            to_name="Client",
            subject="Facture",
            html_content="<p>Facture</p>",
            attachments=[{"filename": "facture.pdf", "content": pdf_content}],
        )

        assert result.success is True
        assert result.message_id == "msg-456"

        # Vérifier que la pièce jointe est envoyée en base64
        payload = mock_post.call_args[1]["json"]
        assert "attachment" in payload
        assert len(payload["attachment"]) == 1
        assert payload["attachment"][0]["name"] == "facture.pdf"
        assert "content" in payload["attachment"][0]  # Base64

    @patch("services.email.brevo_provider.requests.post")
    def test_send_invoice_email_failure(self, mock_post):
        """Test envoi email échoué."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.json.return_value = {"code": "invalid_parameter"}
        mock_post.return_value = mock_response

        provider = BrevoEmailProvider(api_key="test_key")
        result = provider.send_invoice_email(
            from_email="invalid",
            from_name="Test",
            to_email="client@example.com",
            to_name="Client",
            subject="Test",
            html_content="<p>Test</p>",
        )

        assert result.success is False
        assert result.message_id is None
        assert "400" in result.error

    @patch("services.email.brevo_provider.requests.post")
    def test_send_invoice_email_network_error(self, mock_post):
        """Test envoi email avec erreur réseau."""
        mock_post.side_effect = Exception("Network error")

        provider = BrevoEmailProvider(api_key="test_key")
        result = provider.send_invoice_email(
            from_email="noreply@test.ch",
            from_name="Test",
            to_email="client@example.com",
            to_name="Client",
            subject="Test",
            html_content="<p>Test</p>",
        )

        assert result.success is False
        assert "Network error" in result.error

    @patch("services.email.brevo_provider.requests.get")
    def test_verify_domain_verified(self, mock_get):
        """Test vérification domaine vérifié."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "domains": [
                {
                    "domain_name": "test.ch",
                    "verified": True,
                    "dns_records": {
                        "spf_record": "v=spf1 include:spf.brevo.com ~all",
                        "dkim_record": "k=rsa; p=MIGfMA...",
                    },
                }
            ]
        }
        mock_get.return_value = mock_response

        provider = BrevoEmailProvider(api_key="test_key")
        result = provider.verify_domain("test.ch")

        assert result.verified is True
        assert result.domain == "test.ch"
        assert "spf.brevo.com" in result.spf_record
        assert "k=rsa" in result.dkim_record

    @patch("services.email.brevo_provider.requests.get")
    def test_verify_domain_not_verified(self, mock_get):
        """Test vérification domaine non vérifié."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "domains": [
                {
                    "domain_name": "test.ch",
                    "verified": False,
                    "dns_records": {
                        "spf_record": "v=spf1 include:spf.brevo.com ~all",
                        "dkim_record": "k=rsa; p=MIGfMA...",
                    },
                }
            ]
        }
        mock_get.return_value = mock_response

        provider = BrevoEmailProvider(api_key="test_key")
        result = provider.verify_domain("test.ch")

        assert result.verified is False
        assert result.domain == "test.ch"
        assert result.spf_record is not None
        assert result.dkim_record is not None

    @patch("services.email.brevo_provider.requests.get")
    def test_verify_domain_not_found(self, mock_get):
        """Test vérification domaine non configuré."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"domains": []}
        mock_get.return_value = mock_response

        provider = BrevoEmailProvider(api_key="test_key")
        result = provider.verify_domain("notfound.ch")

        assert result.verified is False
        assert result.domain == "notfound.ch"
        assert "non configuré" in result.error

    @patch("services.email.brevo_provider.requests.get")
    def test_get_domain_dns_records(self, mock_get):
        """Test récupération des enregistrements DNS."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "domains": [
                {
                    "domain_name": "test.ch",
                    "verified": False,
                    "dns_records": {
                        "spf_record": "v=spf1 include:spf.brevo.com ~all",
                        "dkim_record": "k=rsa; p=MIGfMA...",
                    },
                }
            ]
        }
        mock_get.return_value = mock_response

        provider = BrevoEmailProvider(api_key="test_key")
        dns_records = provider.get_domain_dns_records("test.ch")

        assert dns_records is not None
        assert "spf" in dns_records
        assert "dkim" in dns_records
        assert "spf.brevo.com" in dns_records["spf"]

    @patch("services.email.brevo_provider.requests.get")
    def test_test_connection_success(self, mock_get):
        """Test connexion réussie."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        provider = BrevoEmailProvider(api_key="test_key")
        assert provider.test_connection() is True

    @patch("services.email.brevo_provider.requests.get")
    def test_test_connection_failure(self, mock_get):
        """Test connexion échouée."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        provider = BrevoEmailProvider(api_key="test_key")
        assert provider.test_connection() is False


class TestEmailResult:
    """Tests de la classe EmailResult."""

    def test_email_result_success(self):
        """Test résultat succès."""
        result = EmailResult(success=True, message_id="msg-123")
        assert result.success is True
        assert result.message_id == "msg-123"
        assert result.error is None

    def test_email_result_failure(self):
        """Test résultat échec."""
        result = EmailResult(success=False, error="Erreur test")
        assert result.success is False
        assert result.message_id is None
        assert result.error == "Erreur test"


class TestDomainVerificationResult:
    """Tests de la classe DomainVerificationResult."""

    def test_domain_verification_verified(self):
        """Test domaine vérifié."""
        result = DomainVerificationResult(
            verified=True,
            domain="test.ch",
            spf_record="v=spf1 include:spf.brevo.com ~all",
            dkim_record="k=rsa; p=MIGfMA...",
        )
        assert result.verified is True
        assert result.domain == "test.ch"
        assert result.spf_record is not None
        assert result.dkim_record is not None

    def test_domain_verification_not_verified(self):
        """Test domaine non vérifié."""
        result = DomainVerificationResult(
            verified=False, domain="test.ch", error="DNS non configuré"
        )
        assert result.verified is False
        assert result.error == "DNS non configuré"
