"""Tests pour recompute_invoice_totals().

Ces tests garantissent que la fonction de recalcul des totaux de facture:
- Calcule correctement les totaux à partir des invoice_lines
- Met à jour subtotal_amount, total_amount, balance_due
- Gère correctement les factures inexistantes
- Est idempotente (peut être appelée plusieurs fois)

Contexte:
- Bug corrigé: des factures avaient total_amount=0 alors que des
  invoice_lines existaient avec des montants corrects.
- Fix: fonction de recalcul centralisée appelée avant génération PDF.
"""

import uuid
from decimal import Decimal

import pytest

from ext import db as _db
from infrastructure.invoices.invoice_calculator import (
    recompute_invoice_totals,
    round_to_5_cents,
)
from models import Client, Company, User
from models.enums import InvoiceBillingStrategy, InvoiceStatus, UserRole
from models.invoice import Invoice, InvoiceLine


class TestRecomputeInvoiceTotals:
    """Tests pour recompute_invoice_totals()."""

    @pytest.fixture
    def sample_company(self, db):
        """Crée une company de test."""
        unique_suffix = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"company_{unique_suffix}"
        user.email = f"company-{unique_suffix}@test.ch"
        user.role = UserRole.company
        user.public_id = str(uuid.uuid4())
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.flush()

        company = Company()
        company.name = "Test Transport SA"
        company.address = "Rue Test 1, 1000 Lausanne"
        company.contact_phone = "0211234567"
        company.contact_email = f"contact_{unique_suffix}@test.ch"
        company.user_id = user.id
        db.session.add(company)
        db.session.flush()
        return company

    @pytest.fixture
    def sample_client(self, db, sample_company):
        """Crée un client de test."""
        unique_suffix = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"client_{unique_suffix}"
        user.email = f"client-{unique_suffix}@test.ch"
        user.role = UserRole.client
        user.first_name = "Jean"
        user.last_name = "Dupont"
        user.phone = "0791234567"
        user.address = "Rue Client 1, 1000 Lausanne"
        user.public_id = str(uuid.uuid4())
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.flush()

        client = Client()
        client.user_id = user.id
        client.company_id = sample_company.id
        client.billing_address = "Rue Client 1, 1000 Lausanne"
        client.contact_email = user.email
        client.contact_phone = "0791234567"
        db.session.add(client)
        db.session.flush()
        return client

    @pytest.fixture
    def invoice_with_zero_totals(self, db, sample_company, sample_client):
        """Crée une facture avec totaux à 0 mais des lignes existantes."""
        invoice = Invoice()
        invoice.company_id = sample_company.id
        invoice.client_id = sample_client.id
        invoice.invoice_number = f"TEST-RECOMPUTE-{uuid.uuid4().hex[:8]}"
        invoice.status = InvoiceStatus.DRAFT
        invoice.billing_strategy = InvoiceBillingStrategy.S1_PATIENT
        # Totaux initialisés à 0
        invoice.subtotal_amount = Decimal("0.00")
        invoice.vat_total_amount = Decimal("0.00")
        invoice.total_amount = Decimal("0.00")
        invoice.balance_due = Decimal("0.00")
        invoice.period_month = 1
        invoice.period_year = 2026
        db.session.add(invoice)
        db.session.flush()

        # Ajouter des lignes de facture
        lines_data = [
            ("Transport A → B", Decimal("40.00")),
            ("Transport B → C", Decimal("40.00")),
            ("Transport C → D [A/R]", Decimal("80.00")),
        ]
        for desc, amount in lines_data:
            line = InvoiceLine()
            line.invoice_id = invoice.id
            line.description = desc
            line.qty = Decimal("1")
            line.unit_price = amount
            line.line_total = amount
            line.vat_rate = Decimal("0.00")
            line.vat_amount = Decimal("0.00")
            line.total_with_vat = amount
            db.session.add(line)

        db.session.flush()
        return invoice

    def test_recompute_updates_totals_from_lines(self, db, invoice_with_zero_totals):
        """Test: recalcul met à jour les totaux à partir des lignes."""
        invoice = invoice_with_zero_totals

        # Vérifier état initial
        assert invoice.total_amount == Decimal("0.00")
        assert invoice.subtotal_amount == Decimal("0.00")

        # Act
        result = recompute_invoice_totals(invoice.id, commit=True)
        db.session.refresh(invoice)

        # Assert
        assert result is not None
        assert result["lines_count"] == 3
        assert invoice.subtotal_amount == Decimal("160.00")
        assert invoice.total_amount == Decimal("160.00")
        assert invoice.balance_due == Decimal("160.00")
        assert result["total"] == Decimal("160.00")

    def test_recompute_returns_none_for_nonexistent_invoice(self, db):
        """Test: retourne None si la facture n'existe pas."""
        result = recompute_invoice_totals(999999999, commit=False)
        assert result is None

    def test_recompute_is_idempotent(self, db, invoice_with_zero_totals):
        """Test: appeler plusieurs fois donne le même résultat."""
        invoice = invoice_with_zero_totals

        # Premier appel
        result1 = recompute_invoice_totals(invoice.id, commit=True)
        db.session.refresh(invoice)
        total_after_first = invoice.total_amount

        # Deuxième appel
        result2 = recompute_invoice_totals(invoice.id, commit=True)
        db.session.refresh(invoice)
        total_after_second = invoice.total_amount

        # Assert
        assert total_after_first == total_after_second
        assert result1["total"] == result2["total"]
        assert invoice.total_amount == Decimal("160.00")

    def test_recompute_handles_empty_invoice(self, db, sample_company, sample_client):
        """Test: facture sans lignes → totaux restent à 0."""
        # Créer une facture sans lignes
        invoice = Invoice()
        invoice.company_id = sample_company.id
        invoice.client_id = sample_client.id
        invoice.invoice_number = f"TEST-EMPTY-{uuid.uuid4().hex[:8]}"
        invoice.status = InvoiceStatus.DRAFT
        invoice.billing_strategy = InvoiceBillingStrategy.S1_PATIENT
        invoice.subtotal_amount = Decimal("0.00")
        invoice.vat_total_amount = Decimal("0.00")
        invoice.total_amount = Decimal("0.00")
        invoice.balance_due = Decimal("0.00")
        invoice.period_month = 1
        invoice.period_year = 2026
        db.session.add(invoice)
        db.session.flush()

        # Act
        result = recompute_invoice_totals(invoice.id, commit=True)
        db.session.refresh(invoice)

        # Assert
        assert result is not None
        assert result["lines_count"] == 0
        assert invoice.total_amount == Decimal("0.00")
        assert invoice.subtotal_amount == Decimal("0.00")

    def test_recompute_with_vat(self, db, sample_company, sample_client):
        """Test: recalcul avec TVA."""
        # Créer une facture avec TVA
        invoice = Invoice()
        invoice.company_id = sample_company.id
        invoice.client_id = sample_client.id
        invoice.invoice_number = f"TEST-VAT-{uuid.uuid4().hex[:8]}"
        invoice.status = InvoiceStatus.DRAFT
        invoice.billing_strategy = InvoiceBillingStrategy.S1_PATIENT
        invoice.subtotal_amount = Decimal("0.00")
        invoice.vat_total_amount = Decimal("0.00")
        invoice.total_amount = Decimal("0.00")
        invoice.balance_due = Decimal("0.00")
        invoice.period_month = 1
        invoice.period_year = 2026
        db.session.add(invoice)
        db.session.flush()

        # Ajouter une ligne avec TVA
        line = InvoiceLine()
        line.invoice_id = invoice.id
        line.description = "Transport avec TVA"
        line.qty = Decimal("1")
        line.unit_price = Decimal("100.00")
        line.line_total = Decimal("100.00")
        line.vat_rate = Decimal("8.1")
        line.vat_amount = Decimal("8.10")
        line.total_with_vat = Decimal("108.10")
        db.session.add(line)
        db.session.flush()

        # Act
        result = recompute_invoice_totals(invoice.id, commit=True)
        db.session.refresh(invoice)

        # Assert
        assert result is not None
        assert invoice.subtotal_amount == Decimal("100.00")
        # Le total est arrondi à 5 centimes (108.10 → 108.10)
        assert invoice.total_amount == Decimal("108.10")

    def test_recompute_without_commit_is_preview(self, db, invoice_with_zero_totals):
        """Test: commit=False est un vrai mode preview (pas de modification)."""
        invoice = invoice_with_zero_totals
        original_total = invoice.total_amount

        # Act avec commit=False (preview)
        result = recompute_invoice_totals(invoice.id, commit=False)

        # Assert: résultat retourné mais facture NON modifiée
        assert result is not None
        assert result["total"] == Decimal("160.00")
        assert result["preview"] is True

        # La facture ne doit PAS avoir été modifiée
        db.session.refresh(invoice)
        assert invoice.total_amount == original_total, (
            "commit=False should NOT modify the invoice (preview mode)"
        )

    def test_recompute_respects_existing_payments(
        self, db, sample_company, sample_client
    ):
        """Test: balance_due tient compte de amount_paid existant."""
        # Créer une facture avec un paiement partiel
        invoice = Invoice()
        invoice.company_id = sample_company.id
        invoice.client_id = sample_client.id
        invoice.invoice_number = f"TEST-PAID-{uuid.uuid4().hex[:8]}"
        invoice.status = InvoiceStatus.DRAFT
        invoice.billing_strategy = InvoiceBillingStrategy.S1_PATIENT
        invoice.subtotal_amount = Decimal("0.00")
        invoice.vat_total_amount = Decimal("0.00")
        invoice.total_amount = Decimal("0.00")
        invoice.balance_due = Decimal("0.00")
        invoice.amount_paid = Decimal("50.00")  # Paiement partiel existant
        invoice.period_month = 1
        invoice.period_year = 2026
        db.session.add(invoice)
        db.session.flush()

        # Ajouter une ligne
        line = InvoiceLine()
        line.invoice_id = invoice.id
        line.description = "Transport"
        line.qty = Decimal("1")
        line.unit_price = Decimal("100.00")
        line.line_total = Decimal("100.00")
        line.vat_rate = Decimal("0.00")
        line.vat_amount = Decimal("0.00")
        line.total_with_vat = Decimal("100.00")
        db.session.add(line)
        db.session.flush()

        # Act
        result = recompute_invoice_totals(invoice.id, commit=True)
        db.session.refresh(invoice)

        # Assert: balance_due = total - amount_paid
        assert invoice.total_amount == Decimal("100.00")
        assert invoice.balance_due == Decimal("50.00")  # 100 - 50
        assert result["balance_due"] == Decimal("50.00")


def test_round_to_5_cents_spec_examples() -> None:
    """Arrondi total facture / QR : multiples de 0,05 CHF (demi au supérieur)."""
    assert round_to_5_cents(Decimal("10.32")) == Decimal("10.30")
    assert round_to_5_cents(Decimal("10.33")) == Decimal("10.35")
    assert round_to_5_cents(Decimal("10.36")) == Decimal("10.35")
    assert round_to_5_cents(Decimal("10.38")) == Decimal("10.40")
