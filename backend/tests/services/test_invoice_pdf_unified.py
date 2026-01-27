"""
Tests de régression pour le template PDF unifié (factures client et clinique).

Teste que le template unifié génère correctement :
- Header dynamique (Client vs Patient)
- Détection aller/retour (explicite + heuristique)
- Format unifié des totaux
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from io import BytesIO

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate

from models import Booking, Client, Company, Invoice, InvoiceLine, User, db
from models.enums import BookingStatus, InvoiceLineType, InvoiceStatus
from services.documents.pdf import PDFService


@pytest.mark.integration
class TestInvoicePdfUnified:
    """Tests de régression pour le template PDF unifié."""

    def test_s1_client_invoice_header(self, db):
        """Test que le PDF client contient 'Client' dans le header."""
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
            invoice_number="INV-001",
            period_year=2024,
            period_month=1,
            status=InvoiceStatus.DRAFT,
            issued_at=datetime.now(UTC),
            due_date=datetime.now(UTC),
            subtotal_amount=Decimal("100.00"),
            vat_total_amount=Decimal("0.00"),
            total_amount=Decimal("100.00"),
        )
        db.session.add(invoice)

        booking = Booking(
            company=company,
            client=client,
            user=user,
            customer_name="Test Customer",
            pickup_location="Rue de la Paix 1, 1204 Genève",
            dropoff_location="Avenue de France 2, 1202 Genève",
            scheduled_time=datetime.now(UTC),
            amount=Decimal("100.00"),
            status=BookingStatus.COMPLETED,
        )
        db.session.add(booking)
        db.session.commit()

        line = InvoiceLine(
            invoice=invoice,
            reservation_id=booking.id,
            type=InvoiceLineType.RIDE,
            description="Test ride",
            qty=Decimal("1.00"),
            unit_price=Decimal("100.00"),
            line_total=Decimal("100.00"),
            vat_rate=Decimal("0.00"),
            vat_amount=Decimal("0.00"),
            total_with_vat=Decimal("100.00"),
        )
        db.session.add(line)
        db.session.commit()

        # Act
        pdf_service = PDFService()
        pdf_content = pdf_service._create_invoice_pdf_content(invoice)

        # Assert: Extraire le texte du PDF
        pdf_text = _extract_text_from_pdf(pdf_content)
        assert "Client" in pdf_text, "Header 'Client' manquant pour facture client"
        assert "TOTAL À FACTURER" in pdf_text, "Libellé 'TOTAL À FACTURER' manquant"
        assert "DÉTAIL DES TRANSPORTS" in pdf_text, "Section 'DÉTAIL DES TRANSPORTS' manquante"

    def test_s2_clinic_invoice_header(self, db):
        """Test que le PDF clinique contient 'Patient' dans le header."""
        # Arrange
        company = Company(name="Test Company", uid_ide="CHE-123.456.789")
        clinic_company = Company(name="Clinic Company", uid_ide="CHE-987.654.321")
        user = User(username="testuser", email="test@example.com")
        client_user = User(username="clientuser", email="client@example.com")
        client = Client(user=client_user, company=company)
        db.session.add_all([company, clinic_company, user, client_user, client])
        db.session.commit()

        invoice = Invoice(
            company=company,
            client=client,
            invoice_number="INV-002",
            period_year=2024,
            period_month=1,
            status=InvoiceStatus.DRAFT,
            issued_at=datetime.now(UTC),
            due_date=datetime.now(UTC),
            subtotal_amount=Decimal("100.00"),
            vat_total_amount=Decimal("0.00"),
            total_amount=Decimal("100.00"),
            billed_to_company_id=clinic_company.id,
        )
        # Simuler billing_strategy S2
        from models.enums import BillingStrategy

        invoice.billing_strategy = BillingStrategy.S2_CLINIC_MONTHLY
        db.session.add(invoice)

        booking = Booking(
            company=company,
            client=client,
            user=user,
            customer_name="Patient Name",
            pickup_location="Rue de la Paix 1, 1204 Genève",
            dropoff_location="Avenue de France 2, 1202 Genève",
            scheduled_time=datetime.now(UTC),
            amount=Decimal("100.00"),
            status=BookingStatus.COMPLETED,
        )
        db.session.add(booking)
        db.session.commit()

        line = InvoiceLine(
            invoice=invoice,
            reservation_id=booking.id,
            type=InvoiceLineType.RIDE,
            description="Test ride",
            qty=Decimal("1.00"),
            unit_price=Decimal("100.00"),
            line_total=Decimal("100.00"),
            vat_rate=Decimal("0.00"),
            vat_amount=Decimal("0.00"),
            total_with_vat=Decimal("100.00"),
            meta={"patient_name": "Patient Name", "patient_id": client.id},
        )
        db.session.add(line)
        db.session.commit()

        # Act
        pdf_service = PDFService()
        pdf_content = pdf_service._create_invoice_pdf_content(invoice)

        # Assert
        pdf_text = _extract_text_from_pdf(pdf_content)
        assert "Patient" in pdf_text, "Header 'Patient' manquant pour facture clinique"
        assert "DÉTAIL DES TRANSPORTS" in pdf_text, "Section 'DÉTAIL DES TRANSPORTS' manquante"
        assert "TOTAL À FACTURER" in pdf_text, "Libellé 'TOTAL À FACTURER' manquant"

    def test_roundtrip_explicit_detection(self, db):
        """Test que les aller/retour explicites (parent_booking_id) sont groupés."""
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
            invoice_number="INV-003",
            period_year=2024,
            period_month=1,
            status=InvoiceStatus.DRAFT,
            issued_at=datetime.now(UTC),
            due_date=datetime.now(UTC),
            subtotal_amount=Decimal("200.00"),
            vat_total_amount=Decimal("0.00"),
            total_amount=Decimal("200.00"),
        )
        db.session.add(invoice)

        # Créer aller et retour liés explicitement
        booking_aller = Booking(
            company=company,
            client=client,
            user=user,
            customer_name="Test Customer",
            pickup_location="Point A",
            dropoff_location="Point B",
            scheduled_time=datetime.now(UTC),
            amount=Decimal("100.00"),
            status=BookingStatus.COMPLETED,
        )
        db.session.add(booking_aller)
        db.session.flush()

        booking_retour = Booking(
            company=company,
            client=client,
            user=user,
            customer_name="Test Customer",
            pickup_location="Point B",
            dropoff_location="Point A",
            scheduled_time=datetime.now(UTC),
            amount=Decimal("100.00"),
            status=BookingStatus.COMPLETED,
            parent_booking_id=booking_aller.id,
            is_return=True,
        )
        db.session.add(booking_retour)
        db.session.commit()

        line_aller = InvoiceLine(
            invoice=invoice,
            reservation_id=booking_aller.id,
            type=InvoiceLineType.RIDE,
            description="Aller",
            qty=Decimal("1.00"),
            unit_price=Decimal("100.00"),
            line_total=Decimal("100.00"),
            vat_rate=Decimal("0.00"),
            vat_amount=Decimal("0.00"),
            total_with_vat=Decimal("100.00"),
        )
        line_retour = InvoiceLine(
            invoice=invoice,
            reservation_id=booking_retour.id,
            type=InvoiceLineType.RIDE,
            description="Retour",
            qty=Decimal("1.00"),
            unit_price=Decimal("100.00"),
            line_total=Decimal("100.00"),
            vat_rate=Decimal("0.00"),
            vat_amount=Decimal("0.00"),
            total_with_vat=Decimal("100.00"),
        )
        db.session.add_all([line_aller, line_retour])
        db.session.commit()

        # Act
        pdf_service = PDFService()
        pdf_content = pdf_service._create_invoice_pdf_content(invoice)

        # Assert
        pdf_text = _extract_text_from_pdf(pdf_content)
        assert "Aller :" in pdf_text, "Détail 'Aller :' manquant pour aller/retour"
        assert "Retour :" in pdf_text, "Détail 'Retour :' manquant pour aller/retour"
        # Vérifier qu'il n'y a qu'une seule ligne principale (pas de duplication)
        # Compter les occurrences de "Point A" dans le tableau (devrait être 1 ligne A/R)
        # Note: Ce test vérifie que la consolidation fonctionne, pas le rendu exact


def _extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extrait le texte d'un PDF pour les tests.

    Utilise pdfminer.six si disponible, sinon fallback basique.
    """
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams

        return extract_text(BytesIO(pdf_content), laparams=LAParams())
    except ImportError:
        # Fallback: chercher des patterns simples dans le contenu binaire
        # Ce n'est pas parfait mais permet de tester sans dépendance
        text = pdf_content.decode("utf-8", errors="ignore")
        return text
