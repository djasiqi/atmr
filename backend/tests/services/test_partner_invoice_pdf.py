"""
Smoke tests pour la génération PDF des factures partenaires.

Teste que le template partenaire génère correctement :
- 2 pages (contenu + QR-Bill)
- Footer fixe en bas de page 1
- QR-Bill sur page 2 avec "Section paiement" et "Récépissé"
- Référence SCOR (RF...) présente
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import BytesIO

import pytest

from models import Booking, Client, Company, CompanyBillingSettings, User
from models.booking_transfer import BookingTransfer
from models.enums import BookingStatus, PartnershipStatus, TransferModel, TransferStatus
from models.partner_invoice import PartnerInvoice, PartnerInvoiceStatus
from models.partnership import Partnership
from services.partnerships.invoices_pdf import generate_partner_invoice_pdf_content


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
        return pdf_content.decode("utf-8", errors="ignore")


def _get_pdf_page_count(pdf_content: bytes) -> int:
    """Compte le nombre de pages dans un PDF.

    Utilise pypdf si disponible, sinon fallback basique.
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_content))
        return len(reader.pages)
    except ImportError:
        try:
            # Fallback: PyPDF2 (ancien nom)
            from PyPDF2 import PdfReader

            reader = PdfReader(BytesIO(pdf_content))
            return len(reader.pages)
        except ImportError:
            # Dernier fallback: compter les occurrences de "/Type /Page"
            # Ce n'est pas parfait mais donne une approximation
            content_str = pdf_content.decode("latin-1", errors="ignore")
            return content_str.count("/Type /Page") - content_str.count("/Type /Pages")


def _create_test_partner_invoice_setup(db, num_transfers: int = 1):
    """Crée les données de test pour une facture partenaire.

    Args:
        db: Session de base de données
        num_transfers: Nombre de transferts à créer

    Returns:
        Tuple (partner_invoice, transfers)
    """
    # Créer les entreprises
    owner_company = Company(
        name="Emmenez Moi",
        uid_ide="CHE-273.048.653",
        address="Rue Verte 8, 1205, Genève",
        contact_email="info@emmenez-moi.ch",
        contact_phone="022 512 02 03",
    )
    partner_company = Company(
        name="MobileEnVille",
        uid_ide="CHE-123.456.789",
        address="Chemin de la Caroline 18, 1213, Petit-Lancy",
        contact_email="info@mobileenville.ch",
        contact_phone="022 870 10 77",
    )
    db.session.add_all([owner_company, partner_company])
    db.session.flush()

    # Créer les billing settings pour l'entreprise exécutante
    billing_settings = CompanyBillingSettings(
        company_id=owner_company.id,
        iban="CH6509000000152631289",
        payment_terms_days=30,
        overdue_fee=Decimal("5.00"),
    )
    db.session.add(billing_settings)

    # Créer le partenariat
    partnership = Partnership(
        owner_company_id=owner_company.id,
        partner_company_id=partner_company.id,
        status=PartnershipStatus.ACCEPTED,
        default_transfer_model=TransferModel.SUBCONTRACT,
        payment_terms_days=30,
    )
    db.session.add(partnership)
    db.session.flush()

    # Créer un client et un utilisateur pour les bookings
    user = User(username="testuser", email="test@example.com")
    client_user = User(username="clientuser", email="client@example.com")
    db.session.add_all([user, client_user])
    db.session.flush()

    client = Client(user=client_user, company=owner_company)
    db.session.add(client)
    db.session.flush()

    # Créer les bookings et les transferts
    transfers = []
    for i in range(num_transfers):
        booking = Booking(
            company=owner_company,
            client=client,
            user=user,
            customer_name=f"Client Test {i + 1}",
            pickup_location=f"Départ {i + 1}, 1200 Genève",
            dropoff_location=f"Arrivée {i + 1}, 1205 Genève",
            scheduled_time=datetime.now(UTC) - timedelta(days=i),
            amount=Decimal("40.00"),
            status=BookingStatus.COMPLETED,
        )
        db.session.add(booking)
        db.session.flush()

        transfer = BookingTransfer(
            booking_id=booking.id,
            partnership_id=partnership.id,
            transfer_model=TransferModel.SUBCONTRACT,
            status=TransferStatus.COMPLETED,
            is_validated=True,
            validated_at=datetime.now(UTC) - timedelta(days=i),
            partner_cost=Decimal("40.00"),
            currency="CHF",
            executing_company_id=owner_company.id,
            requesting_company_id=partner_company.id,
        )
        db.session.add(transfer)
        transfers.append(transfer)

    db.session.flush()

    # Créer la facture partenaire
    total_amount = Decimal("40.00") * num_transfers
    partner_invoice = PartnerInvoice(
        partnership_id=partnership.id,
        executing_company_id=owner_company.id,
        period_year=2026,
        period_month=1,
        invoice_number=f"PARTNER-EM-2026-01-{num_transfers:04d}",
        subtotal_amount=total_amount,
        vat_amount=Decimal("0.00"),
        total_amount=total_amount,
        currency="CHF",
        status=PartnerInvoiceStatus.DRAFT,
        issued_at=datetime.now(UTC),
        due_date=datetime.now(UTC) + timedelta(days=30),
    )
    db.session.add(partner_invoice)
    db.session.commit()

    # Recharger les objets après commit
    partner_invoice = PartnerInvoice.query.get(partner_invoice.id)
    transfers = [BookingTransfer.query.get(t.id) for t in transfers]

    return partner_invoice, transfers


@pytest.mark.integration
class TestPartnerInvoicePdf:
    """Smoke tests pour la génération PDF des factures partenaires."""

    def test_partner_invoice_pdf_1_transfer_no_exception(self, db):
        """Test que la génération PDF avec 1 transfert ne lève pas d'exception."""
        # Arrange
        partner_invoice, transfers = _create_test_partner_invoice_setup(db, num_transfers=1)

        # Act & Assert: pas d'exception
        pdf_content = generate_partner_invoice_pdf_content(partner_invoice, transfers)
        assert pdf_content is not None
        assert len(pdf_content) > 0

    def test_partner_invoice_pdf_30_transfers_no_exception(self, db):
        """Test que la génération PDF avec 30 transferts ne lève pas d'exception."""
        # Arrange
        partner_invoice, transfers = _create_test_partner_invoice_setup(db, num_transfers=30)

        # Act & Assert: pas d'exception
        pdf_content = generate_partner_invoice_pdf_content(partner_invoice, transfers)
        assert pdf_content is not None
        assert len(pdf_content) > 0

    def test_partner_invoice_pdf_has_2_pages(self, db):
        """Test que le PDF contient exactement 2 pages (contenu + QR-Bill)."""
        # Arrange
        partner_invoice, transfers = _create_test_partner_invoice_setup(db, num_transfers=1)

        # Act
        pdf_content = generate_partner_invoice_pdf_content(partner_invoice, transfers)

        # Assert
        page_count = _get_pdf_page_count(pdf_content)
        assert page_count == 2, f"Le PDF devrait avoir 2 pages, mais en a {page_count}"

    def test_partner_invoice_pdf_30_transfers_has_at_least_2_pages(self, db):
        """Test que le PDF avec 30 transferts contient au moins 2 pages."""
        # Arrange
        partner_invoice, transfers = _create_test_partner_invoice_setup(db, num_transfers=30)

        # Act
        pdf_content = generate_partner_invoice_pdf_content(partner_invoice, transfers)

        # Assert
        page_count = _get_pdf_page_count(pdf_content)
        assert page_count >= 2, f"Le PDF devrait avoir au moins 2 pages, mais en a {page_count}"

    def test_partner_invoice_pdf_qrbill_section_paiement(self, db):
        """Test que la 2e page contient 'Section paiement' (QR-Bill suisse)."""
        # Arrange
        partner_invoice, transfers = _create_test_partner_invoice_setup(db, num_transfers=1)

        # Act
        pdf_content = generate_partner_invoice_pdf_content(partner_invoice, transfers)
        pdf_text = _extract_text_from_pdf(pdf_content)

        # Assert
        assert "Section paiement" in pdf_text, (
            "Le QR-Bill devrait contenir 'Section paiement'"
        )

    def test_partner_invoice_pdf_qrbill_recepisse(self, db):
        """Test que la 2e page contient 'Récépissé' (QR-Bill suisse)."""
        # Arrange
        partner_invoice, transfers = _create_test_partner_invoice_setup(db, num_transfers=1)

        # Act
        pdf_content = generate_partner_invoice_pdf_content(partner_invoice, transfers)
        pdf_text = _extract_text_from_pdf(pdf_content)

        # Assert
        # Note: "Récépissé" peut être écrit "Récépissé" ou "Recepisse" selon l'encodage
        assert "piss" in pdf_text.lower() or "Récépissé" in pdf_text, (
            "Le QR-Bill devrait contenir 'Récépissé'"
        )

    def test_partner_invoice_pdf_contains_scor_reference(self, db):
        """Test que le QR-Bill contient une référence SCOR (RF...)."""
        # Arrange
        partner_invoice, transfers = _create_test_partner_invoice_setup(db, num_transfers=1)

        # Act
        pdf_content = generate_partner_invoice_pdf_content(partner_invoice, transfers)
        pdf_text = _extract_text_from_pdf(pdf_content)

        # Assert: La référence SCOR commence par "RF"
        assert "RF" in pdf_text, (
            "Le QR-Bill devrait contenir une référence SCOR (RF...)"
        )

    def test_partner_invoice_pdf_contains_footer_message(self, db):
        """Test que le PDF contient le message de footer (conditions de paiement)."""
        # Arrange
        partner_invoice, transfers = _create_test_partner_invoice_setup(db, num_transfers=1)

        # Act
        pdf_content = generate_partner_invoice_pdf_content(partner_invoice, transfers)
        pdf_text = _extract_text_from_pdf(pdf_content)

        # Assert
        assert "règlement" in pdf_text.lower() or "paiement" in pdf_text.lower(), (
            "Le footer devrait contenir les conditions de paiement"
        )

    def test_partner_invoice_pdf_contains_invoice_number(self, db):
        """Test que le PDF contient le numéro de facture."""
        # Arrange
        partner_invoice, transfers = _create_test_partner_invoice_setup(db, num_transfers=1)

        # Act
        pdf_content = generate_partner_invoice_pdf_content(partner_invoice, transfers)
        pdf_text = _extract_text_from_pdf(pdf_content)

        # Assert
        assert partner_invoice.invoice_number in pdf_text, (
            f"Le PDF devrait contenir le numéro de facture '{partner_invoice.invoice_number}'"
        )

    def test_partner_invoice_pdf_contains_iban(self, db):
        """Test que le PDF contient l'IBAN."""
        # Arrange
        partner_invoice, transfers = _create_test_partner_invoice_setup(db, num_transfers=1)

        # Act
        pdf_content = generate_partner_invoice_pdf_content(partner_invoice, transfers)
        pdf_text = _extract_text_from_pdf(pdf_content)

        # Assert: L'IBAN ou une partie de l'IBAN doit être présent
        # CH65 0900 0000 1526 3128 9 (formaté) ou CH6509000000152631289 (non formaté)
        assert "CH65" in pdf_text or "1526 3128" in pdf_text, (
            "Le PDF devrait contenir l'IBAN"
        )

    def test_partner_invoice_pdf_contains_total_amount(self, db):
        """Test que le PDF contient le montant total."""
        # Arrange
        partner_invoice, transfers = _create_test_partner_invoice_setup(db, num_transfers=1)

        # Act
        pdf_content = generate_partner_invoice_pdf_content(partner_invoice, transfers)
        pdf_text = _extract_text_from_pdf(pdf_content)

        # Assert
        total_str = f"{partner_invoice.total_amount:.2f}"
        assert total_str in pdf_text or "40.00" in pdf_text, (
            f"Le PDF devrait contenir le montant total '{total_str}'"
        )
