"""
Tests pour vérifier que la facture initiale reste INTACTE lors de la génération d'un rappel.

Objectif: S'assurer que:
- invoice.pdf_url reste inchangé
- invoice.total_amount reste inchangé
- invoice.due_date reste inchangé
- invoice.lines reste inchangé
- Le rappel a son propre PDF (reminder.pdf_url) distinct de invoice.pdf_url
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import BytesIO
from pathlib import Path

import pytest

from application.invoices.generate_invoice_reminder import (
    GenerateInvoiceReminderInput,
    GenerateInvoiceReminderUseCase,
)
from models import (
    Company,
    CompanyBillingSettings,
    Invoice,
    InvoiceLine,
    InvoiceReminder,
    User,
)
from models.enums import InvoiceLineType, InvoiceStatus
from services.documents.pdf import PDFService


def _ensure_company_billing_settings(db, company_id: int) -> None:
    """GenerateInvoiceReminderUseCase exige des paramètres de facturation."""
    if CompanyBillingSettings.query.filter_by(company_id=company_id).first():
        return
    db.session.add(CompanyBillingSettings(company_id=company_id))
    db.session.flush()


def _extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extrait le texte d'un PDF pour les tests.

    Utilise pdfminer.six si disponible, sinon fallback basique.
    """
    try:
        from pdfminer.high_level import extract_text
        from pdfminer.layout import LAParams

        return extract_text(BytesIO(pdf_content), laparams=LAParams())
    except ImportError:
        return pdf_content.decode("utf-8", errors="ignore")


@pytest.mark.integration
class TestReminderPdfInvoiceIntact:
    """Tests pour vérifier que la facture initiale reste intacte lors de la génération d'un rappel."""

    def test_generate_reminder_does_not_modify_invoice_pdf_url(
        self, db, sample_company, sample_client
    ):
        """Test que invoice.pdf_url reste inchangé après génération d'un rappel."""
        if not all([sample_company, sample_client]):
            pytest.skip("Required fixtures missing")

        _ensure_company_billing_settings(db, sample_company.id)

        # Arrange: Créer une facture avec un PDF initial
        invoice = Invoice(
            company=sample_company,
            client=sample_client,
            invoice_number="INV-TEST-001",
            period_year=datetime.now(UTC).year,
            period_month=datetime.now(UTC).month,
            status=InvoiceStatus.OVERDUE,
            issued_at=datetime.now(UTC) - timedelta(days=20),
            due_date=datetime.now(UTC) - timedelta(days=10),
            subtotal_amount=Decimal("100.00"),
            vat_total_amount=Decimal("0.00"),
            total_amount=Decimal("100.00"),
            pdf_url="/uploads/invoices/invoice_INV-TEST-001_20240101_120000.pdf",
            reminder_level=0,
        )
        db.session.add(invoice)

        # Ajouter une ligne de facture
        line = InvoiceLine(
            invoice=invoice,
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

        # Capturer l'état initial de la facture
        invoice_id = invoice.id
        original_pdf_url = invoice.pdf_url
        original_total = invoice.total_amount
        original_due_date = invoice.due_date
        original_lines_count = len(invoice.lines)
        original_reminder_level = invoice.reminder_level

        # Act: Générer un rappel niveau 1
        use_case = GenerateInvoiceReminderUseCase()
        result = use_case.execute(GenerateInvoiceReminderInput(invoice_id=invoice_id, level=1))

        # Assert: Vérifier que le rappel a été créé
        assert result.success, "La génération du rappel a échoué"
        assert result.reminder is not None, "Le rappel n'a pas été créé"
        assert result.reminder.pdf_url is not None, "Le rappel n'a pas de PDF"

        # Assert: Vérifier que la facture initiale est INTACTE
        db.session.refresh(invoice)
        assert invoice.pdf_url == original_pdf_url, (
            f"invoice.pdf_url a été modifié ! "
            f"Attendu: {original_pdf_url}, "
            f"Reçu: {invoice.pdf_url}"
        )
        assert invoice.total_amount == original_total, (
            f"invoice.total_amount a été modifié ! "
            f"Attendu: {original_total}, "
            f"Reçu: {invoice.total_amount}"
        )
        assert invoice.due_date == original_due_date, (
            f"invoice.due_date a été modifié ! "
            f"Attendu: {original_due_date}, "
            f"Reçu: {invoice.due_date}"
        )
        assert len(invoice.lines) == original_lines_count, (
            f"invoice.lines a été modifié ! "
            f"Attendu: {original_lines_count} lignes, "
            f"Reçu: {len(invoice.lines)} lignes"
        )

        # Assert: Vérifier que le PDF du rappel est DISTINCT du PDF de la facture
        assert result.reminder.pdf_url != invoice.pdf_url, (
            f"Le PDF du rappel ne doit pas être le même que celui de la facture ! "
            f"reminder.pdf_url: {result.reminder.pdf_url}, "
            f"invoice.pdf_url: {invoice.pdf_url}"
        )
        assert "reminder_" in result.reminder.pdf_url, (
            f"Le PDF du rappel doit avoir un filename commençant par 'reminder_' ! "
            f"Reçu: {result.reminder.pdf_url}"
        )

        # Assert: Vérifier que reminder_level a été mis à jour (c'est le seul champ qui doit changer)
        assert invoice.reminder_level > original_reminder_level, (
            f"reminder_level devrait être mis à jour ! "
            f"Attendu: > {original_reminder_level}, "
            f"Reçu: {invoice.reminder_level}"
        )

    def test_generate_multiple_reminders_creates_distinct_pdfs(self, db, sample_company, sample_client):
        """Test que plusieurs rappels génèrent des PDFs distincts."""
        if not all([sample_company, sample_client]):
            pytest.skip("Required fixtures missing")

        # Arrange: Créer une facture
        invoice = Invoice(
            company=sample_company,
            client=sample_client,
            invoice_number="INV-TEST-002",
            period_year=datetime.now(UTC).year,
            period_month=datetime.now(UTC).month,
            status=InvoiceStatus.OVERDUE,
            issued_at=datetime.now(UTC) - timedelta(days=30),
            due_date=datetime.now(UTC) - timedelta(days=20),
            subtotal_amount=Decimal("200.00"),
            vat_total_amount=Decimal("0.00"),
            total_amount=Decimal("200.00"),
            pdf_url="/uploads/invoices/invoice_INV-TEST-002_20240101_120000.pdf",
            reminder_level=0,
        )
        db.session.add(invoice)
        db.session.commit()

        original_pdf_url = invoice.pdf_url
        invoice_id = invoice.id

        # Act: Générer 3 rappels successifs
        use_case = GenerateInvoiceReminderUseCase()
        reminder_urls = []

        for level in [1, 2, 3]:
            result = use_case.execute(GenerateInvoiceReminderInput(invoice_id=invoice_id, level=level))
            assert result.success, f"La génération du rappel niveau {level} a échoué"
            assert result.reminder is not None, f"Le rappel niveau {level} n'a pas été créé"
            assert result.reminder.pdf_url is not None, f"Le rappel niveau {level} n'a pas de PDF"
            reminder_urls.append(result.reminder.pdf_url)

            # Vérifier que invoice.pdf_url reste inchangé après chaque rappel
            db.session.refresh(invoice)
            assert invoice.pdf_url == original_pdf_url, (
                f"invoice.pdf_url a été modifié après le rappel niveau {level} !"
            )

        # Assert: Vérifier que tous les PDFs sont distincts
        assert len(set(reminder_urls)) == 3, (
            f"Les 3 rappels doivent avoir des PDFs distincts ! "
            f"Reçu: {reminder_urls}"
        )
        assert invoice.pdf_url not in reminder_urls, (
            "Le PDF de la facture ne doit pas être dans les PDFs des rappels !"
        )

        # Assert: Vérifier que tous les PDFs commencent par "reminder_"
        for url in reminder_urls:
            assert "reminder_" in url, (
                f"Le PDF du rappel doit avoir un filename commençant par 'reminder_' ! "
                f"Reçu: {url}"
            )

    def test_generate_reminder_pdf_service_does_not_modify_invoice(self, db, sample_company, sample_client):
        """Test que PDFService.generate_reminder_pdf ne modifie pas l'invoice."""
        if not all([sample_company, sample_client]):
            pytest.skip("Required fixtures missing")

        _ensure_company_billing_settings(db, sample_company.id)

        # Arrange: Créer une facture
        invoice = Invoice(
            company=sample_company,
            client=sample_client,
            invoice_number="INV-TEST-003",
            period_year=datetime.now(UTC).year,
            period_month=datetime.now(UTC).month,
            status=InvoiceStatus.OVERDUE,
            issued_at=datetime.now(UTC) - timedelta(days=15),
            due_date=datetime.now(UTC) - timedelta(days=5),
            subtotal_amount=Decimal("150.00"),
            vat_total_amount=Decimal("0.00"),
            total_amount=Decimal("150.00"),
            pdf_url="/uploads/invoices/invoice_INV-TEST-003_20240101_120000.pdf",
            reminder_level=0,
        )
        db.session.add(invoice)
        db.session.commit()

        # Capturer l'état initial
        original_pdf_url = invoice.pdf_url
        original_total = invoice.total_amount
        original_due_date = invoice.due_date

        # Act: Appeler directement generate_reminder_pdf
        pdf_service = PDFService()
        reminder_pdf_url = pdf_service.generate_reminder_pdf(invoice, level=1, reminder=None)

        # Assert: Vérifier que l'invoice n'a pas été modifié
        db.session.refresh(invoice)
        assert invoice.pdf_url == original_pdf_url, (
            f"PDFService.generate_reminder_pdf a modifié invoice.pdf_url ! "
            f"Attendu: {original_pdf_url}, Reçu: {invoice.pdf_url}"
        )
        assert invoice.total_amount == original_total, (
            "PDFService.generate_reminder_pdf a modifié invoice.total_amount !"
        )
        assert invoice.due_date == original_due_date, (
            "PDFService.generate_reminder_pdf a modifié invoice.due_date !"
        )

        # Assert: Vérifier que le PDF du rappel est distinct
        assert reminder_pdf_url is not None, "Le PDF du rappel n'a pas été généré"
        assert reminder_pdf_url != invoice.pdf_url, (
            "Le PDF du rappel ne doit pas être le même que celui de la facture !"
        )
        assert "reminder_" in reminder_pdf_url, (
            "Le PDF du rappel doit avoir un filename commençant par 'reminder_' !"
        )

    def test_reminder_pdf_uses_invoice_template(self, db, sample_company, sample_client):
        """Test que le PDF rappel utilise le template facture avec DÉTAIL DES PRESTATIONS."""
        if not all([sample_company, sample_client]):
            pytest.skip("Required fixtures missing")

        _ensure_company_billing_settings(db, sample_company.id)

        # Arrange: Créer une facture avec des lignes
        invoice = Invoice(
            company=sample_company,
            client=sample_client,
            invoice_number="INV-TEST-004",
            period_year=datetime.now(UTC).year,
            period_month=datetime.now(UTC).month,
            status=InvoiceStatus.OVERDUE,
            issued_at=datetime.now(UTC) - timedelta(days=15),
            due_date=datetime.now(UTC) - timedelta(days=5),
            subtotal_amount=Decimal("150.00"),
            vat_total_amount=Decimal("0.00"),
            total_amount=Decimal("150.00"),
            pdf_url="/uploads/invoices/invoice_INV-TEST-004_20240101_120000.pdf",
            reminder_level=0,
        )
        db.session.add(invoice)

        # Ajouter une ligne de facture
        line = InvoiceLine(
            invoice=invoice,
            type=InvoiceLineType.RIDE,
            description="Test ride",
            qty=Decimal("1.00"),
            unit_price=Decimal("150.00"),
            line_total=Decimal("150.00"),
            vat_rate=Decimal("0.00"),
            vat_amount=Decimal("0.00"),
            total_with_vat=Decimal("150.00"),
        )
        db.session.add(line)
        db.session.commit()

        invoice_id = invoice.id
        original_pdf_url = invoice.pdf_url

        # Act: Générer un rappel niveau 1
        use_case = GenerateInvoiceReminderUseCase()
        result = use_case.execute(GenerateInvoiceReminderInput(invoice_id=invoice_id, level=1))

        # Assert: Vérifier que le rappel a été créé
        assert result.success, "La génération du rappel a échoué"
        assert result.reminder is not None, "Le rappel n'a pas été créé"
        assert result.reminder.pdf_url is not None, "Le rappel n'a pas de PDF"

        # Assert: Vérifier que invoice.pdf_url reste inchangé
        db.session.refresh(invoice)
        assert invoice.pdf_url == original_pdf_url, "invoice.pdf_url a été modifié !"

        # Assert: Extraire le texte du PDF rappel et vérifier qu'il contient les sections facture
        pdf_service = PDFService()
        reminder_pdf_content, _ = pdf_service._create_invoice_pdf_content(
            invoice,
            reminder_level=1,
            reminder_fee=result.reminder.reminder_fee_amount or Decimal("0.00"),
            reminder_total_due=result.reminder.total_due or invoice.total_amount,
            reminder_principal=result.reminder.principal_amount,
        )
        pdf_text = _extract_text_from_pdf(reminder_pdf_content)

        # Vérifier que le PDF rappel contient les sections du template facture
        # Marqueur stable et spécifique du template facture (évite flakiness "TOTAL" vs "TOTAL À FACTURER")
        assert "DÉTAIL DES PRESTATIONS" in pdf_text, (
            "Le PDF rappel doit contenir 'DÉTAIL DES PRESTATIONS' (template facture)"
        )
        assert "RAPPEL N°1" in pdf_text or "RAPPEL N° 1" in pdf_text, (
            "Le PDF rappel doit contenir 'RAPPEL N°1'"
        )

        # Vérifier que le PDF rappel contient la ligne de frais de rappel
        if result.reminder.reminder_fee_amount and result.reminder.reminder_fee_amount > 0:
            assert "Frais de rappel" in pdf_text, (
                "Le PDF rappel doit contenir 'Frais de rappel' si des frais sont appliqués"
            )
            # Vérifier que le montant des frais est présent
            fee_str = f"{float(result.reminder.reminder_fee_amount):.2f}"
            assert fee_str in pdf_text, (
                f"Le PDF rappel doit contenir le montant des frais ({fee_str})"
            )
