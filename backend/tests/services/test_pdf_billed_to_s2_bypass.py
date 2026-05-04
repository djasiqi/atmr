"""Tests pour le bypass S2/clinic dans _get_billed_to().

Ces tests garantissent que les factures cliniques (S2_CLINIC_MONTHLY) ou
les factures avec billing_party de type CLINIC/EMS/HOSPITAL utilisent
toujours le billing_party pour "Facturé à", même sans lien ClientBillingParty.

Contexte:
- Bug corrigé: pour les factures multi-patients (S2), le code vérifiait
  un lien ClientBillingParty qui n'a pas de sens (le client_id est arbitraire).
- Fix: bypass de la vérification du lien pour les factures S2/clinic.
"""

import uuid
from decimal import Decimal

import pytest

from ext import db as _db
from models import Client, Company, User
from models.billing_party import BillingParty
from models.enums import (
    BillingPartyType,
    InvoiceBillingStrategy,
    InvoiceStatus,
    UserRole,
)
from models.invoice import Invoice
from services.documents.pdf import _get_billed_to


class TestGetBilledToS2Bypass:
    """Tests pour le bypass S2/clinic dans _get_billed_to()."""

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
        """Crée un client de test (représente un patient)."""
        unique_suffix = str(uuid.uuid4())[:8]
        user = User()
        user.username = f"patient_{unique_suffix}"
        user.email = f"patient-{unique_suffix}@test.ch"
        user.role = UserRole.client
        user.first_name = "Hagai"
        user.last_name = "LUGASSY"
        user.phone = "0791234567"
        user.address = "Rue de Moillebeau 25, 1209 Genève"
        user.public_id = str(uuid.uuid4())
        user.set_password("password123", force_change=False)
        db.session.add(user)
        db.session.flush()

        client = Client()
        client.user_id = user.id
        client.company_id = sample_company.id
        client.billing_address = "Rue de Moillebeau 25, 1209 Genève"
        client.domicile_address = "Rue de Moillebeau 25"
        client.domicile_zip = "1209"
        client.domicile_city = "Genève"
        client.contact_email = user.email
        client.contact_phone = "0791234567"
        db.session.add(client)
        db.session.flush()
        return client

    @pytest.fixture
    def clinic_billing_party(self, db, sample_company):
        """Crée un billing_party de type CLINIC."""
        bp = BillingParty()
        bp.display_name = "Clinique les Hauts d'Anières"
        bp.type = BillingPartyType.CLINIC
        bp.billing_address = "Chemin des Courbes 9, 1247 Anières, Suisse"
        bp.company_id = sample_company.id
        db.session.add(bp)
        db.session.flush()
        return bp

    @pytest.fixture
    def ems_billing_party(self, db, sample_company):
        """Crée un billing_party de type EMS."""
        bp = BillingParty()
        bp.display_name = "EMS Les Tilleuls"
        bp.type = BillingPartyType.EMS
        bp.billing_address = "Route de l'EMS 10, 1200 Genève"
        bp.company_id = sample_company.id
        db.session.add(bp)
        db.session.flush()
        return bp

    @pytest.fixture
    def other_billing_party(self, db, sample_company):
        """Crée un billing_party de type OTHER (tiers payeur standard)."""
        bp = BillingParty()
        bp.display_name = "Tiers Payeur Test"
        bp.type = BillingPartyType.OTHER
        bp.billing_address = "Adresse Tiers Payeur, 1000 Test"
        bp.company_id = sample_company.id
        db.session.add(bp)
        db.session.flush()
        return bp

    def test_get_billed_to_s2_clinic_monthly_without_link(
        self, db, sample_company, sample_client, clinic_billing_party
    ):
        """Test: facture S2_CLINIC_MONTHLY sans lien ClientBillingParty.

        Attendu: "Facturé à" = clinique (pas le patient).
        """
        # Arrange: créer une facture S2 sans lien ClientBillingParty
        invoice = Invoice()
        invoice.company_id = sample_company.id
        invoice.client_id = sample_client.id
        invoice.billing_party_id = clinic_billing_party.id
        invoice.billing_strategy = InvoiceBillingStrategy.S2_CLINIC_MONTHLY
        invoice.invoice_number = f"TEST-S2-{uuid.uuid4().hex[:8]}"
        invoice.status = InvoiceStatus.DRAFT
        invoice.subtotal_amount = Decimal("100.00")
        invoice.vat_total_amount = Decimal("0.00")
        invoice.total_amount = Decimal("100.00")
        invoice.balance_due = Decimal("100.00")
        invoice.period_month = 1
        invoice.period_year = 2026
        db.session.add(invoice)
        db.session.flush()

        # Important: PAS de lien ClientBillingParty créé

        # Act
        name, addr = _get_billed_to(invoice)

        # Assert: le nom doit être la clinique, pas le patient
        assert "Clinique" in name or "ANIÈRES" in name.upper(), (
            f"Expected clinic name, got: {name}"
        )
        assert "LUGASSY" not in name.upper(), (
            f"Patient name should NOT appear in billed_to: {name}"
        )
        assert "Chemin des Courbes" in addr or "1247" in addr, (
            f"Expected clinic address, got: {addr}"
        )

    def test_get_billed_to_clinic_type_without_link(
        self, db, sample_company, sample_client, clinic_billing_party
    ):
        """Test: facture avec billing_party type=CLINIC sans lien.

        Même sans billing_strategy=S2, le type CLINIC doit bypasser le lien.
        """
        # Arrange: facture avec billing_party CLINIC mais pas S2
        invoice = Invoice()
        invoice.company_id = sample_company.id
        invoice.client_id = sample_client.id
        invoice.billing_party_id = clinic_billing_party.id
        invoice.billing_strategy = InvoiceBillingStrategy.S1_PATIENT  # Pas S2
        invoice.invoice_number = f"TEST-CLINIC-{uuid.uuid4().hex[:8]}"
        invoice.status = InvoiceStatus.DRAFT
        invoice.subtotal_amount = Decimal("50.00")
        invoice.vat_total_amount = Decimal("0.00")
        invoice.total_amount = Decimal("50.00")
        invoice.balance_due = Decimal("50.00")
        invoice.period_month = 1
        invoice.period_year = 2026
        db.session.add(invoice)
        db.session.flush()

        # Act
        name, _addr = _get_billed_to(invoice)

        # Assert: doit utiliser la clinique (bypass via type=CLINIC)
        assert "Clinique" in name or "ANIÈRES" in name.upper(), (
            f"Expected clinic name, got: {name}"
        )

    def test_get_billed_to_ems_type_without_link(
        self, db, sample_company, sample_client, ems_billing_party
    ):
        """Test: facture avec billing_party type=EMS sans lien.

        Le type EMS doit aussi bypasser le lien ClientBillingParty.
        """
        # Arrange
        invoice = Invoice()
        invoice.company_id = sample_company.id
        invoice.client_id = sample_client.id
        invoice.billing_party_id = ems_billing_party.id
        invoice.billing_strategy = InvoiceBillingStrategy.S1_PATIENT
        invoice.invoice_number = f"TEST-EMS-{uuid.uuid4().hex[:8]}"
        invoice.status = InvoiceStatus.DRAFT
        invoice.subtotal_amount = Decimal("60.00")
        invoice.vat_total_amount = Decimal("0.00")
        invoice.total_amount = Decimal("60.00")
        invoice.balance_due = Decimal("60.00")
        invoice.period_month = 1
        invoice.period_year = 2026
        db.session.add(invoice)
        db.session.flush()

        # Act
        name, _addr = _get_billed_to(invoice)

        # Assert: doit utiliser l'EMS
        assert "EMS" in name or "Tilleuls" in name, f"Expected EMS name, got: {name}"
        assert "LUGASSY" not in name.upper(), f"Patient name should NOT appear: {name}"

    def test_get_billed_to_other_type_falls_back_to_client_when_no_link(
        self, db, sample_company, sample_client, other_billing_party
    ):
        """Test: facture avec billing_party type=OTHER sans lien.

        Pour les types non-établissement (OTHER, FAMILY, etc.), le fallback
        au client doit s'appliquer si le lien ClientBillingParty manque.
        """
        # Arrange
        invoice = Invoice()
        invoice.company_id = sample_company.id
        invoice.client_id = sample_client.id
        invoice.billing_party_id = other_billing_party.id
        invoice.billing_strategy = InvoiceBillingStrategy.S1_PATIENT
        invoice.invoice_number = f"TEST-OTHER-{uuid.uuid4().hex[:8]}"
        invoice.status = InvoiceStatus.DRAFT
        invoice.subtotal_amount = Decimal("40.00")
        invoice.vat_total_amount = Decimal("0.00")
        invoice.total_amount = Decimal("40.00")
        invoice.balance_due = Decimal("40.00")
        invoice.period_month = 1
        invoice.period_year = 2026
        db.session.add(invoice)
        db.session.flush()

        # Act
        name, _addr = _get_billed_to(invoice)

        # Assert: doit fallback au client car pas de lien et type=OTHER
        assert "LUGASSY" in name.upper(), (
            f"Expected patient name (fallback), got: {name}"
        )
