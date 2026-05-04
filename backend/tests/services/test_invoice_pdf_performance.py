"""
Tests de monitoring performance pour la génération PDF des factures.

Teste que les logs de performance sont émis correctement avec les seuils.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from models import Booking, Client, Company, Invoice, InvoiceLine, User
from models.enums import BookingStatus, InvoiceLineType, InvoiceStatus
from services.documents.pdf import (
    PERF_WARNING_MS_THRESHOLD,
    PERF_WARNING_ROWS_THRESHOLD,
    TEMPLATE_VERSION,
    PDFService,
)


@pytest.mark.integration
class TestInvoicePdfPerformance:
    """Tests de monitoring performance pour la génération PDF."""

    def test_performance_logging_normal(self, db):
        """Test que les logs INFO sont émis pour une facture normale."""
        # Arrange
        company = Company(name="Test Company", uid_ide="CHE-123.456.789")
        user = User(username="testuser", email="test@example.com")
        client_user = User(username="clientuser", email="client@example.com")
        client = Client(user=client_user, company=company)
        db.session.add_all([company, user, client_user, client])
        db.session.commit()

        invoice = Invoice(
            company=company,
            client=client,
            invoice_number="INV-PERF-001",
            period_year=2024,
            period_month=1,
            status=InvoiceStatus.DRAFT,
            issued_at=datetime.now(UTC),
            due_date=datetime.now(UTC),
            subtotal_amount=Decimal("50.00"),
            vat_total_amount=Decimal("0.00"),
            total_amount=Decimal("50.00"),
        )
        db.session.add(invoice)

        # Créer quelques bookings (moins que le seuil)
        for i in range(5):
            booking = Booking(
                company=company,
                client=client,
                user=user,
                customer_name=f"Customer {i}",
                pickup_location=f"Point A{i}",
                dropoff_location=f"Point B{i}",
                scheduled_time=datetime.now(UTC),
                amount=Decimal("10.00"),
                status=BookingStatus.COMPLETED,
            )
            db.session.add(booking)
            line = InvoiceLine(
                invoice=invoice,
                reservation_id=booking.id,
                type=InvoiceLineType.RIDE,
                description=f"Ride {i}",
                qty=Decimal("1.00"),
                unit_price=Decimal("10.00"),
                line_total=Decimal("10.00"),
                vat_rate=Decimal("0.00"),
                vat_amount=Decimal("0.00"),
                total_with_vat=Decimal("10.00"),
            )
            db.session.add(line)
        db.session.commit()

        # Act & Assert
        with patch("services.documents.pdf.app_logger") as mock_logger:
            pdf_service = PDFService()
            pdf_service.generate_invoice_pdf(invoice)

            # Vérifier que INFO a été appelé avec les bonnes données
            assert mock_logger.info.called, "INFO log should be called"
            call_args = mock_logger.info.call_args[0]
            assert "InvoicePDF generated" in call_args[0]
            assert "invoice_id=" in call_args[0]
            assert "nb_rows=" in call_args[0]
            assert "generation_ms=" in call_args[0]
            assert f"template_version={TEMPLATE_VERSION}" in call_args[0]

            # Vérifier que WARNING n'a pas été appelé (facture normale)
            assert not mock_logger.warning.called, (
                "WARNING should not be called for normal invoice"
            )

    def test_performance_logging_large_invoice(self, db):
        """Test que les logs WARNING sont émis pour une facture avec beaucoup de lignes."""
        # Arrange
        company = Company(name="Test Company", uid_ide="CHE-123.456.789")
        user = User(username="testuser", email="test@example.com")
        client_user = User(username="clientuser", email="client@example.com")
        client = Client(user=client_user, company=company)
        db.session.add_all([company, user, client_user, client])
        db.session.commit()

        invoice = Invoice(
            company=company,
            client=client,
            invoice_number="INV-PERF-002",
            period_year=2024,
            period_month=1,
            status=InvoiceStatus.DRAFT,
            issued_at=datetime.now(UTC),
            due_date=datetime.now(UTC),
            subtotal_amount=Decimal("500.00"),
            vat_total_amount=Decimal("0.00"),
            total_amount=Decimal("500.00"),
        )
        db.session.add(invoice)

        # Créer beaucoup de bookings (plus que le seuil)
        nb_bookings = PERF_WARNING_ROWS_THRESHOLD + 10
        for i in range(nb_bookings):
            booking = Booking(
                company=company,
                client=client,
                user=user,
                customer_name=f"Customer {i}",
                pickup_location=f"Point A{i}",
                dropoff_location=f"Point B{i}",
                scheduled_time=datetime.now(UTC),
                amount=Decimal("10.00"),
                status=BookingStatus.COMPLETED,
            )
            db.session.add(booking)
            line = InvoiceLine(
                invoice=invoice,
                reservation_id=booking.id,
                type=InvoiceLineType.RIDE,
                description=f"Ride {i}",
                qty=Decimal("1.00"),
                unit_price=Decimal("10.00"),
                line_total=Decimal("10.00"),
                vat_rate=Decimal("0.00"),
                vat_amount=Decimal("0.00"),
                total_with_vat=Decimal("10.00"),
            )
            db.session.add(line)
        db.session.commit()

        # Act & Assert
        with patch("services.documents.pdf.app_logger") as mock_logger:
            pdf_service = PDFService()
            pdf_service.generate_invoice_pdf(invoice)

            # Vérifier que WARNING a été appelé
            assert mock_logger.warning.called, (
                "WARNING log should be called for large invoice"
            )
            call_args = mock_logger.warning.call_args[0]
            assert "InvoicePDF slow/large" in call_args[0]
            assert "invoice_id=" in call_args[0]
            assert "nb_rows=" in call_args[0]
            assert "generation_ms=" in call_args[0]
            assert f"template_version={TEMPLATE_VERSION}" in call_args[0]
            assert "thresholds_exceeded=" in call_args[0]
            assert "rows=" in call_args[0] or "time=" in call_args[0]
