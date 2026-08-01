"""Bloc « Facturé à » d'une facture patient : domicile du patient concerné.

Régression : tous les résidents d'une même institution recevaient la même adresse,
car le PDF déduisait le patient depuis la dernière demande de transport de
l'institution au lieu de la facture en cours.
"""

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from models import Client, Company, User
from models.billing_party import BillingParty
from models.enums import (
    BillingPartyType,
    InvoiceBillingStrategy,
    InvoiceStatus,
    UserRole,
)
from models.institution import Institution
from models.institution_patient import InstitutionPatient
from models.invoice import Invoice
from services.documents.pdf import _get_billed_to


@pytest.fixture
def company(db):
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"company_{suffix}"
    user.email = f"company-{suffix}@test.ch"
    user.role = UserRole.company
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()

    company = Company()
    company.name = "Test Transport SA"
    company.address = "Rue Test 1, 1000 Lausanne"
    company.contact_phone = "0211234567"
    company.contact_email = f"contact_{suffix}@test.ch"
    company.user_id = user.id
    db.session.add(company)
    db.session.flush()
    return company


@pytest.fixture
def institution(db):
    inst = Institution()
    inst.public_id = str(uuid.uuid4())
    inst.name = f"Clinique les Hauts d'Anières {uuid.uuid4().hex[:6]}"
    inst.institution_type = "clinic"
    inst.address = "Chemin des Courbes 9, 1247 Anières"
    inst.billing_address = "Chemin des Courbes 9, 1247 Anières"
    db.session.add(inst)
    db.session.flush()
    return inst


@pytest.fixture
def shared_institution_client(db, company, institution):
    """Compte client technique partagé par tous les résidents."""
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"institution_{suffix}"
    user.email = f"institution-{suffix}@test.ch"
    user.role = UserRole.client
    user.first_name = "Clinique"
    user.last_name = "ANIERES"
    user.address = "Chemin des Courbes 9, 1247 Anières"
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()

    client = Client()
    client.user_id = user.id
    client.company_id = company.id
    client.is_institution = True
    client.institution_name = institution.name
    client.linked_institution_id = institution.id
    db.session.add(client)
    db.session.flush()
    return client


def _patient(db, institution, *, first, last, street, postal, city):
    patient = InstitutionPatient()
    patient.institution_id = institution.id
    patient.first_name = first
    patient.last_name = last
    patient.address = street
    patient.postal_code = postal
    patient.city = city
    db.session.add(patient)
    db.session.flush()
    return patient


def _patient_billing_party(db, company, patient, *, billing_address):
    bp = BillingParty()
    bp.company_id = company.id
    bp.type = BillingPartyType.PATIENT
    bp.display_name = f"{patient.first_name} {patient.last_name}"
    bp.billing_address = billing_address
    bp.external_ref = f"patient:{patient.id}"
    bp.is_active = True
    db.session.add(bp)
    db.session.flush()
    return bp


def _invoice(db, company, client, patient, *, billing_party=None):
    invoice = Invoice()
    invoice.company_id = company.id
    invoice.client_id = client.id
    invoice.institution_patient_id = patient.id
    invoice.billing_party_id = billing_party.id if billing_party else None
    invoice.billing_strategy = InvoiceBillingStrategy.S1_PATIENT
    invoice.invoice_number = f"TEST-{uuid.uuid4().hex[:8]}"
    invoice.status = InvoiceStatus.DRAFT
    invoice.subtotal_amount = Decimal("45.00")
    invoice.vat_total_amount = Decimal("0.00")
    invoice.total_amount = Decimal("45.00")
    invoice.balance_due = Decimal("45.00")
    invoice.period_month = 7
    invoice.period_year = 2026
    invoice.due_date = datetime(2026, 8, 31, tzinfo=UTC)
    db.session.add(invoice)
    db.session.flush()
    return invoice


@pytest.fixture
def chatellier(db, institution):
    return _patient(
        db,
        institution,
        first="Philippe",
        last="CHATELLIER",
        street="Chemin du Gué 69",
        postal="1213",
        city="Petit-Lancy",
    )


@pytest.fixture
def camoletti(db, institution):
    return _patient(
        db,
        institution,
        first="Eva",
        last="CAMOLETTI",
        street="Route de Thonon 14",
        postal="1222",
        city="Vésenaz",
    )


def test_each_patient_keeps_own_domicile(
    db, company, shared_institution_client, chatellier, camoletti
):
    """Deux résidents de la même institution ne partagent pas une adresse."""
    bp_a = _patient_billing_party(
        db, company, chatellier, billing_address="Chemin du Gué 69\n1213 Petit-Lancy"
    )
    bp_b = _patient_billing_party(
        db, company, camoletti, billing_address="Route de Thonon 14\n1222 Vésenaz"
    )
    invoice_a = _invoice(
        db, company, shared_institution_client, chatellier, billing_party=bp_a
    )
    invoice_b = _invoice(
        db, company, shared_institution_client, camoletti, billing_party=bp_b
    )

    name_a, addr_a = _get_billed_to(invoice_a)
    name_b, addr_b = _get_billed_to(invoice_b)

    assert "CHATELLIER" in name_a.upper()
    assert "Chemin du Gué 69" in addr_a
    assert "1213 Petit-Lancy" in addr_a
    assert "CAMOLETTI" in name_b.upper()
    assert "Route de Thonon 14" in addr_b
    assert "1222 Vésenaz" in addr_b


def test_patient_domicile_overrides_stale_billing_party_address(
    db, company, shared_institution_client, camoletti
):
    """Un snapshot BillingParty périmé ne doit pas écraser le domicile courant."""
    bp = _patient_billing_party(
        db, company, camoletti, billing_address="Chemin du Gué 69\n1213 Petit-Lancy"
    )
    invoice = _invoice(
        db, company, shared_institution_client, camoletti, billing_party=bp
    )

    _name, addr = _get_billed_to(invoice)

    assert "Route de Thonon 14" in addr
    assert "Chemin du Gué" not in addr


def test_patient_billing_party_used_without_client_link(
    db, company, shared_institution_client, chatellier
):
    """Aucun ClientBillingParty n'existe : le payeur patient reste utilisé."""
    bp = _patient_billing_party(
        db, company, chatellier, billing_address="Chemin du Gué 69\n1213 Petit-Lancy"
    )
    invoice = _invoice(
        db, company, shared_institution_client, chatellier, billing_party=bp
    )

    name, addr = _get_billed_to(invoice)

    assert "CHATELLIER" in name.upper()
    assert "ANIERES" not in name.upper()
    assert "Chemin des Courbes" not in addr


def test_fallback_without_billing_party_uses_invoice_patient(
    db, company, shared_institution_client, camoletti
):
    """Sans BillingParty, l'adresse vient du patient de la facture."""
    invoice = _invoice(db, company, shared_institution_client, camoletti)

    name, addr = _get_billed_to(invoice)

    assert "CAMOLETTI" in name.upper()
    assert "Route de Thonon 14" in addr
    assert "1222 Vésenaz" in addr
