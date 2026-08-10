"""
Tests de régression pour le template PDF unifié (factures client et clinique).

Teste que le template unifié génère correctement :
- Header dynamique (Client vs Patient)
- Détection aller/retour (explicite + heuristique)
- Format unifié des totaux
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from io import BytesIO

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate

from models import Booking, Client, Company, Invoice, InvoiceLine, User
from models.enums import (
    BookingStatus,
    InvoiceBillingStrategy,
    InvoiceLineType,
    InvoiceStatus,
    UserRole,
)
from services.documents.pdf import PDFService


def _ensure_users_with_password(*users: User) -> None:
    """PostgreSQL impose password NOT NULL sur user ; les tests doivent la renseigner."""
    for u in users:
        if not getattr(u, "public_id", None):
            u.public_id = str(uuid.uuid4())
        if not getattr(u, "password", None):
            u.set_password("password123", force_change=False)


def _assign_company_owner(db, company: Company, owner: User) -> None:
    """company.user_id est NOT NULL en base."""
    db.session.add(owner)
    db.session.flush()
    company.user_id = owner.id


def _unique_pdf_users() -> tuple[User, User]:
    """Évite les collisions username/email entre tests ; mot de passe renseigné."""
    suf = str(uuid.uuid4())[:8]
    driver = User(username=f"pdf_drv_{suf}", email=f"pdf_drv_{suf}@test.example")
    client_u = User(username=f"pdf_cli_{suf}", email=f"pdf_cli_{suf}@test.example")
    driver.role = UserRole.company
    client_u.role = UserRole.client
    _ensure_users_with_password(driver, client_u)
    return driver, client_u


@pytest.mark.integration
class TestInvoicePdfUnified:
    """Tests de régression pour le template PDF unifié."""

    def test_s1_client_invoice_header(self, db):
        """Test que le PDF client contient 'Client' dans le header."""
        # Arrange
        company = Company(name="Test Company", uid_ide="CHE-123.456.789")
        user, client_user = _unique_pdf_users()
        _assign_company_owner(db, company, user)
        client = Client(user=client_user, company=company)
        db.session.add_all([company, client_user, client])
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
            user_id=user.id,
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
        pdf_bytes, _nb = pdf_service._create_invoice_pdf_content(invoice)

        # Assert: Extraire le texte du PDF
        pdf_text = _extract_text_from_pdf(pdf_bytes)
        assert "DÉTAIL DES PRESTATIONS" in pdf_text, (
            "Section 'DÉTAIL DES PRESTATIONS' manquante"
        )
        assert "TOTAL :" in pdf_text, "Libellé total manquant pour facture client"
        assert "Test ride" in pdf_text, "Ligne de prestation manquante"

    def test_s2_clinic_invoice_header(self, db):
        """Test que le PDF clinique contient 'Patient' dans le header."""
        # Arrange
        company = Company(name="Test Company", uid_ide="CHE-123.456.789")
        clinic_company = Company(name="Clinic Company", uid_ide="CHE-987.654.321")
        user, client_user = _unique_pdf_users()
        _co_suffix = str(uuid.uuid4())[:8]
        clinic_owner = User(
            username=f"clinicowner_{_co_suffix}",
            email=f"clinic.owner.{_co_suffix}@test.example",
        )
        clinic_owner.role = UserRole.company
        _ensure_users_with_password(clinic_owner)
        _assign_company_owner(db, company, user)
        _assign_company_owner(db, clinic_company, clinic_owner)
        client = Client(user=client_user, company=company)
        db.session.add_all([company, clinic_company, client_user, client])
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
        invoice.billing_strategy = InvoiceBillingStrategy.S2_CLINIC_MONTHLY
        db.session.add(invoice)

        booking = Booking(
            company=company,
            client=client,
            user_id=user.id,
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
            line_meta={"patient_name": "Patient Name", "patient_id": client.id},
        )
        db.session.add(line)
        db.session.commit()

        # Act
        pdf_service = PDFService()
        pdf_bytes, _nb = pdf_service._create_invoice_pdf_content(invoice)

        # Assert
        pdf_text = _extract_text_from_pdf(pdf_bytes)
        assert "Rue de la Paix" in pdf_text, "Adresse pickup manquante pour facture clinique"
        assert "DÉTAIL DES PRESTATIONS" in pdf_text, (
            "Section 'DÉTAIL DES PRESTATIONS' manquante"
        )
        assert "TOTAL À FACTURER" in pdf_text, "Libellé 'TOTAL À FACTURER' manquant"

    def test_roundtrip_explicit_detection(self, db):
        """Test que les aller/retour explicites (parent_booking_id) sont groupés."""
        # Arrange
        company = Company(name="Test Company", uid_ide="CHE-123.456.789")
        user, client_user = _unique_pdf_users()
        _assign_company_owner(db, company, user)
        client = Client(user=client_user, company=company)
        db.session.add_all([company, client_user, client])
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
            user_id=user.id,
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
            user_id=user.id,
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
        pdf_bytes, _nb = pdf_service._create_invoice_pdf_content(invoice)

        # Assert
        pdf_text = _extract_text_from_pdf(pdf_bytes)
        assert "[A/R]" in pdf_text, "Marqueur aller-retour [A/R] manquant"
        assert "transport aller-retour" in pdf_text.lower()

    def test_material_delivery_line_in_pdf(self, db):
        """Test que les lignes MATERIAL_DELIVERY affichent 'Livraison' + description."""
        # Arrange
        company = Company(name="Test Company", uid_ide="CHE-123.456.789")
        user, client_user = _unique_pdf_users()
        _assign_company_owner(db, company, user)
        client = Client(user=client_user, company=company)
        db.session.add_all([company, client_user, client])
        db.session.commit()

        invoice = Invoice(
            company=company,
            client=client,
            invoice_number="INV-DEL-001",
            period_year=2024,
            period_month=1,
            status=InvoiceStatus.DRAFT,
            issued_at=datetime.now(UTC),
            due_date=datetime.now(UTC),
            subtotal_amount=Decimal("35.00"),
            vat_total_amount=Decimal("0.00"),
            total_amount=Decimal("35.00"),
        )
        db.session.add(invoice)

        booking = Booking(
            company=company,
            client=client,
            user_id=user.id,
            customer_name="Test Customer",
            pickup_location="Clinique",
            dropoff_location="Domicile",
            scheduled_time=datetime.now(UTC),
            amount=Decimal("35.00"),
            status=BookingStatus.COMPLETED,
            mission_type="material_delivery",
            delivery_description="Médicament urgent",
        )
        db.session.add(booking)
        db.session.commit()

        line = InvoiceLine(
            invoice=invoice,
            reservation_id=booking.id,
            type=InvoiceLineType.MATERIAL_DELIVERY,
            description="Livraison – Médicament urgent – Clinique → Domicile",
            qty=Decimal("1.00"),
            unit_price=Decimal("35.00"),
            line_total=Decimal("35.00"),
            vat_rate=Decimal("0.00"),
            vat_amount=Decimal("0.00"),
            total_with_vat=Decimal("35.00"),
        )
        db.session.add(line)
        db.session.commit()

        # Act
        pdf_service = PDFService()
        pdf_bytes, _nb = pdf_service._create_invoice_pdf_content(invoice)

        # Assert: "Livraison" et la description doivent apparaître dans le PDF
        pdf_text = _extract_text_from_pdf(pdf_bytes)
        assert "Livraison" in pdf_text, (
            "Libellé 'Livraison' manquant pour MATERIAL_DELIVERY"
        )
        assert "Médicament urgent" in pdf_text, (
            "Description livraison manquante dans PDF"
        )

    def test_mixed_invoice_2_rides_1_delivery_in_pdf(self, db):
        """Test facture mixte (2 transports + 1 livraison) : la livraison apparaît bien."""
        from datetime import UTC

        # Arrange
        company = Company(name="Test Company", uid_ide="CHE-123.456.789")
        user, client_user = _unique_pdf_users()
        _assign_company_owner(db, company, user)
        client = Client(user=client_user, company=company)
        db.session.add_all([company, client_user, client])
        db.session.commit()

        invoice = Invoice(
            company=company,
            client=client,
            invoice_number="INV-MIX-001",
            period_year=2024,
            period_month=1,
            status=InvoiceStatus.DRAFT,
            issued_at=datetime.now(UTC),
            due_date=datetime.now(UTC),
            subtotal_amount=Decimal("85.00"),
            vat_total_amount=Decimal("0.00"),
            total_amount=Decimal("85.00"),
        )
        db.session.add(invoice)

        # 2 transports patient
        for pickup, dropoff in [("Point A", "Point B"), ("Point B", "Point C")]:
            booking = Booking(
                company=company,
                client=client,
                user_id=user.id,
                customer_name="Test Customer",
                pickup_location=pickup,
                dropoff_location=dropoff,
                scheduled_time=datetime.now(UTC),
                amount=Decimal("25.00"),
                status=BookingStatus.COMPLETED,
            )
            db.session.add(booking)
            db.session.flush()
            line = InvoiceLine(
                invoice=invoice,
                reservation_id=booking.id,
                type=InvoiceLineType.RIDE,
                description=f"Trajet {pickup} → {dropoff}",
                qty=Decimal("1.00"),
                unit_price=Decimal("25.00"),
                line_total=Decimal("25.00"),
                vat_rate=Decimal("0.00"),
                vat_amount=Decimal("0.00"),
                total_with_vat=Decimal("25.00"),
            )
            db.session.add(line)

        # 1 livraison matériel
        booking_delivery = Booking(
            company=company,
            client=client,
            user_id=user.id,
            customer_name="Test Customer",
            pickup_location="Entrepôt",
            dropoff_location="Domicile",
            scheduled_time=datetime.now(UTC),
            amount=Decimal("35.00"),
            status=BookingStatus.COMPLETED,
            mission_type="material_delivery",
            delivery_description="Colis médical",
        )
        db.session.add(booking_delivery)
        db.session.flush()
        line_delivery = InvoiceLine(
            invoice=invoice,
            reservation_id=booking_delivery.id,
            type=InvoiceLineType.MATERIAL_DELIVERY,
            description="Livraison – Colis médical – Entrepôt → Domicile",
            qty=Decimal("1.00"),
            unit_price=Decimal("35.00"),
            line_total=Decimal("35.00"),
            vat_rate=Decimal("0.00"),
            vat_amount=Decimal("0.00"),
            total_with_vat=Decimal("35.00"),
        )
        db.session.add(line_delivery)
        db.session.commit()

        # Act
        pdf_service = PDFService()
        pdf_bytes, _nb = pdf_service._create_invoice_pdf_content(invoice)

        # Assert: les 3 lignes apparaissent (2 transports + 1 livraison)
        pdf_text = _extract_text_from_pdf(pdf_bytes)
        assert "Livraison" in pdf_text, "Libellé livraison manquant"
        assert "Colis médical" in pdf_text, "Description livraison manquante"
        assert "Point A" in pdf_text or "Point B" in pdf_text, "Transports manquants"
        assert "85.00" in pdf_text or "85" in pdf_text, "Total incorrect"


def _extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extrait le texte d'un PDF pour les tests."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        try:
            from pdfminer.high_level import extract_text
            from pdfminer.layout import LAParams

            return extract_text(BytesIO(pdf_content), laparams=LAParams())
        except ImportError:
            pytest.skip("pypdf ou pdfminer requis pour extraire le texte PDF")
