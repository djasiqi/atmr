"""
Tests unitaires pour le use case SendInvoiceByEmailUseCase.

Teste l'envoi de factures par email via Brevo avec des mocks.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from application.invoices.send_invoice_by_email import (
    SendInvoiceByEmailInput,
    SendInvoiceByEmailResult,
    SendInvoiceByEmailUseCase,
)
from models.enums import InvoiceStatus
from services.email.base import EmailResult


@pytest.fixture(autouse=True)
def _invoice_email_app_context(app):
    """Model.query / Mock(spec=Model) exigent un contexte Flask."""
    with (
        app.app_context(),
        patch("application.invoices.send_invoice_by_email.BrevoEmailProvider"),
        patch(
            "application.invoices.send_invoice_by_email.inject_signature_into_html",
            side_effect=lambda html, **_kwargs: (html, None),
        ),
        patch(
            "application.invoices.send_invoice_by_email.ensure_draft_pdf_ready_for_send",
            return_value=(True, None),
        ),
    ):
        yield


@pytest.fixture
def mock_invoice():
    """Mock d'une facture (sans spec= : évite les attributs hybrides Mock)."""
    invoice = Mock()
    invoice.id = 1
    invoice.invoice_number = "INV-2026-001"
    invoice.company_id = 1
    invoice.client_id = 1
    invoice.amount_with_vat = Decimal("107.70")
    invoice.total_amount = Decimal("107.70")
    invoice.due_date = datetime(2026, 2, 15)
    invoice.pdf_url = "/uploads/invoices/INV-2026-001.pdf"
    invoice.status = InvoiceStatus.DRAFT
    invoice.meta = {}
    invoice.billing_party = None
    invoice.billed_to_company = None
    invoice.billed_to_type = None
    invoice.bill_to_client = None
    invoice.mark_as_sent = Mock()
    return invoice


@pytest.fixture
def mock_client():
    """Mock d'un client."""
    user = Mock()
    user.first_name = "Test"
    user.last_name = "Client"
    user.username = "testclient"
    client = Mock()
    client.id = 1
    client.name = "Test Client"
    client.contact_email = "client@test.ch"
    client.user = user
    return client


@pytest.fixture
def mock_company():
    """Mock d'une entreprise."""
    company = Mock()
    company.id = 1
    company.name = "Test Company"
    company.meta = None
    company.user = None
    return company


@pytest.fixture
def mock_billing_settings():
    """Mock des paramètres de facturation."""
    settings = Mock()
    settings.smtp_username = "noreply@testcompany.ch"
    settings.from_name = "Test Company"
    settings.domain_verified = True
    settings.invoice_message_template = None
    return settings


class TestSendInvoiceByEmailUseCase:
    """Tests pour SendInvoiceByEmailUseCase."""

    def test_send_invoice_success(
        self, mock_invoice, mock_client, mock_company, mock_billing_settings
    ):
        """Test envoi réussi d'une facture par email."""
        # Setup
        use_case = SendInvoiceByEmailUseCase()

        # Mock des queries
        with (
            patch(
                "application.invoices.send_invoice_by_email.Invoice.query"
            ) as mock_invoice_query,
            patch(
                "application.invoices.send_invoice_by_email.Client.query"
            ) as mock_client_query,
            patch(
                "application.invoices.send_invoice_by_email.Company.query"
            ) as mock_company_query,
            patch(
                "application.invoices.send_invoice_by_email.CompanyBillingSettings.query"
            ) as mock_settings_query,
            patch("application.invoices.send_invoice_by_email.db.session"),
            patch("application.invoices.send_invoice_by_email.Path") as mock_path,
            patch.object(
                use_case.brevo_provider, "send_invoice_email"
            ) as mock_send_email,
        ):
            # Configuration des mocks
            mock_invoice_query.get.return_value = mock_invoice
            mock_client_query.get.return_value = mock_client
            mock_company_query.get.return_value = mock_company
            mock_settings_query.filter_by.return_value.first.return_value = (
                mock_billing_settings
            )

            # Mock du fichier PDF
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.open.return_value.__enter__.return_value.read.return_value = b"fake_pdf_content"
            mock_path.return_value = mock_path_instance

            # Mock de l'envoi email
            mock_send_email.return_value = EmailResult(
                success=True, message_id="brevo-message-123"
            )

            # Exécution
            input_data = SendInvoiceByEmailInput(invoice_id=1)
            result = use_case.execute(input_data)

            # Assertions
            assert result.success is True
            assert result.invoice_id == 1
            assert result.recipient == "client@test.ch"
            assert result.sent_at is not None
            assert result.error is None

            # Vérifier que mark_as_sent a été appelé
            mock_invoice.mark_as_sent.assert_called_once()

            # Vérifier que send_invoice_email a été appelé avec les bons arguments
            mock_send_email.assert_called_once()
            call_args = mock_send_email.call_args
            assert call_args.kwargs["from_email"] == "noreply@testcompany.ch"
            assert call_args.kwargs["from_name"] == "Test Company"
            assert call_args.kwargs["to_email"] == "client@test.ch"
            assert call_args.kwargs["to_name"] == "Test Client"
            assert "INV-2026-001" in call_args.kwargs["subject"]

    def test_send_invoice_not_found(self):
        """Test envoi d'une facture inexistante."""
        use_case = SendInvoiceByEmailUseCase()

        with patch(
            "application.invoices.send_invoice_by_email.Invoice.query"
        ) as mock_invoice_query:
            mock_invoice_query.get.return_value = None

            input_data = SendInvoiceByEmailInput(invoice_id=999)
            result = use_case.execute(input_data)

            assert result.success is False
            assert result.status_code == 404
            assert "introuvable" in result.error.lower()

    def test_send_invoice_client_no_email(
        self, mock_invoice, mock_client, mock_company
    ):
        """Test envoi d'une facture à un client sans email."""
        use_case = SendInvoiceByEmailUseCase()
        mock_client.contact_email = None

        with (
            patch(
                "application.invoices.send_invoice_by_email.Invoice.query"
            ) as mock_invoice_query,
            patch(
                "application.invoices.send_invoice_by_email.Client.query"
            ) as mock_client_query,
            patch(
                "application.invoices.send_invoice_by_email.Company.query"
            ) as mock_company_query,
        ):
            mock_invoice_query.get.return_value = mock_invoice
            mock_client_query.get.return_value = mock_client
            mock_company_query.get.return_value = mock_company

            input_data = SendInvoiceByEmailInput(invoice_id=1)
            result = use_case.execute(input_data)

            assert result.success is False
            assert result.status_code == 400
            assert "email" in result.error.lower()

    def test_send_invoice_domain_not_verified(
        self, mock_invoice, mock_client, mock_company, mock_billing_settings
    ):
        """Test envoi d'une facture avec domaine non vérifié."""
        use_case = SendInvoiceByEmailUseCase()
        mock_billing_settings.domain_verified = False

        with (
            patch(
                "application.invoices.send_invoice_by_email.Invoice.query"
            ) as mock_invoice_query,
            patch(
                "application.invoices.send_invoice_by_email.Client.query"
            ) as mock_client_query,
            patch(
                "application.invoices.send_invoice_by_email.Company.query"
            ) as mock_company_query,
            patch(
                "application.invoices.send_invoice_by_email.CompanyBillingSettings.query"
            ) as mock_settings_query,
        ):
            mock_invoice_query.get.return_value = mock_invoice
            mock_client_query.get.return_value = mock_client
            mock_company_query.get.return_value = mock_company
            mock_settings_query.filter_by.return_value.first.return_value = (
                mock_billing_settings
            )

            input_data = SendInvoiceByEmailInput(invoice_id=1)
            result = use_case.execute(input_data)

            assert result.success is False
            assert result.status_code == 403
            assert "vérifié" in result.error.lower()

    def test_send_invoice_brevo_error(
        self, mock_invoice, mock_client, mock_company, mock_billing_settings
    ):
        """Test envoi d'une facture avec erreur Brevo."""
        use_case = SendInvoiceByEmailUseCase()

        with (
            patch(
                "application.invoices.send_invoice_by_email.Invoice.query"
            ) as mock_invoice_query,
            patch(
                "application.invoices.send_invoice_by_email.Client.query"
            ) as mock_client_query,
            patch(
                "application.invoices.send_invoice_by_email.Company.query"
            ) as mock_company_query,
            patch(
                "application.invoices.send_invoice_by_email.CompanyBillingSettings.query"
            ) as mock_settings_query,
            patch("application.invoices.send_invoice_by_email.db.session"),
            patch("application.invoices.send_invoice_by_email.Path") as mock_path,
            patch.object(
                use_case.brevo_provider, "send_invoice_email"
            ) as mock_send_email,
        ):
            mock_invoice_query.get.return_value = mock_invoice
            mock_client_query.get.return_value = mock_client
            mock_company_query.get.return_value = mock_company
            mock_settings_query.filter_by.return_value.first.return_value = (
                mock_billing_settings
            )

            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.open.return_value.__enter__.return_value.read.return_value = b"fake_pdf_content"
            mock_path.return_value = mock_path_instance

            # Mock d'une erreur Brevo
            mock_send_email.return_value = EmailResult(
                success=False, error="API key invalid"
            )

            input_data = SendInvoiceByEmailInput(invoice_id=1)
            result = use_case.execute(input_data)

            assert result.success is False
            assert result.status_code == 500
            assert "Brevo" in result.error

    def test_send_invoice_custom_recipient(
        self, mock_invoice, mock_client, mock_company, mock_billing_settings
    ):
        """Test envoi d'une facture à un destinataire personnalisé."""
        use_case = SendInvoiceByEmailUseCase()

        with (
            patch(
                "application.invoices.send_invoice_by_email.Invoice.query"
            ) as mock_invoice_query,
            patch(
                "application.invoices.send_invoice_by_email.Client.query"
            ) as mock_client_query,
            patch(
                "application.invoices.send_invoice_by_email.Company.query"
            ) as mock_company_query,
            patch(
                "application.invoices.send_invoice_by_email.CompanyBillingSettings.query"
            ) as mock_settings_query,
            patch("application.invoices.send_invoice_by_email.db.session"),
            patch("application.invoices.send_invoice_by_email.Path") as mock_path,
            patch.object(
                use_case.brevo_provider, "send_invoice_email"
            ) as mock_send_email,
        ):
            mock_invoice_query.get.return_value = mock_invoice
            mock_client_query.get.return_value = mock_client
            mock_company_query.get.return_value = mock_company
            mock_settings_query.filter_by.return_value.first.return_value = (
                mock_billing_settings
            )

            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_instance.open.return_value.__enter__.return_value.read.return_value = b"fake_pdf_content"
            mock_path.return_value = mock_path_instance

            mock_send_email.return_value = EmailResult(
                success=True, message_id="brevo-message-123"
            )

            # Envoi à un destinataire personnalisé
            input_data = SendInvoiceByEmailInput(
                invoice_id=1, recipient_email="custom@example.com"
            )
            result = use_case.execute(input_data)

            assert result.success is True
            assert result.recipient == "custom@example.com"

            # Vérifier que l'email a été envoyé au bon destinataire
            call_args = mock_send_email.call_args
            assert call_args.kwargs["to_email"] == "custom@example.com"
