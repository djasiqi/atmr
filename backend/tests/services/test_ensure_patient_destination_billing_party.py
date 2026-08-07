"""ensure_patient_destination_billing_party — bascule clinique → patient."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from application.invoices.billing_opportunities import pick_canonical_billing_party_id
from models import (
    BillingParty,
    Booking,
    Client,
    Company,
    Institution,
    InstitutionPatient,
    User,
)
from models.enums import BillingPartyType, BookingStatus, UserRole
from services.billing.billing_party_linker import (
    ensure_patient_destination_billing_party,
    get_or_create_billing_party_for_institution_patient,
    is_establishment_billing_party,
    resolve_billing_party_for_portfolio_patient,
)


def _make_company(db, *, suffix: str | None = None) -> Company:
    suffix = suffix or uuid.uuid4().hex[:8]
    user = User()
    user.username = f"co_{suffix}"
    user.email = f"co-{suffix}@test.ch"
    user.role = UserRole.company
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()

    company = Company()
    company.name = f"Transport {suffix}"
    company.address = "Rue Test 1, 1200 Genève"
    company.contact_phone = "0220000000"
    company.contact_email = f"contact-{suffix}@test.ch"
    company.user_id = user.id
    db.session.add(company)
    db.session.flush()
    return company


def _make_client(db, company: Company) -> Client:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"cli_{suffix}"
    user.email = f"cli-{suffix}@test.ch"
    user.role = UserRole.client
    user.first_name = "Martine"
    user.last_name = "BOCHATAY"
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()

    client = Client()
    client.user_id = user.id
    client.company_id = company.id
    client.domicile_address = "Route d'Anières 1"
    client.domicile_zip = "1247"
    client.domicile_city = "Anières"
    client.default_billed_to_type = "patient"
    db.session.add(client)
    db.session.flush()
    return client


def _clinic_bp(db, company: Company) -> BillingParty:
    bp = BillingParty()
    bp.company_id = company.id
    bp.type = BillingPartyType.CLINIC
    bp.display_name = "Clinique les Hauts d'Anières"
    bp.billing_address = "Chemin des Courbes 9\n1247 Anières"
    bp.external_ref = f"clinic_company:{uuid.uuid4().hex[:6]}"
    bp.is_active = True
    db.session.add(bp)
    db.session.flush()
    return bp


@pytest.fixture
def company(db):
    return _make_company(db)


def test_is_establishment_billing_party():
    clinic = BillingParty()
    clinic.type = BillingPartyType.CLINIC
    patient = BillingParty()
    patient.type = BillingPartyType.PATIENT
    assert is_establishment_billing_party(clinic) is True
    assert is_establishment_billing_party(patient) is False
    assert is_establishment_billing_party(None) is False


def test_ensure_replaces_clinic_bp_with_portfolio_patient(db, company):
    client = _make_client(db, company)
    clinic = _clinic_bp(db, company)

    booking = Booking()
    booking.company_id = company.id
    booking.client_id = client.id
    booking.customer_name = "BOCHATAY Martine"
    booking.scheduled_time = datetime(2026, 8, 10, 10, 0, 0)
    booking.status = BookingStatus.COMPLETED.value
    booking.pickup_location = "Anières"
    booking.dropoff_location = "Nyon"
    booking.amount = 40.0
    booking.billed_to_type = "patient"
    booking.billed_to_company_id = 99
    booking.billing_party_id = clinic.id
    booking.booking_type = "manual"
    db.session.add(booking)
    db.session.flush()

    bp = ensure_patient_destination_billing_party(booking)
    db.session.flush()

    assert bp is not None
    assert bp.type == BillingPartyType.PATIENT
    assert bp.id != clinic.id
    assert booking.billing_party_id == bp.id
    assert booking.billed_to_company_id is None
    assert bp.external_ref == f"patient_client:{client.id}"


def test_ensure_keeps_non_establishment_bp(db, company):
    client = _make_client(db, company)
    curator = BillingParty()
    curator.company_id = company.id
    curator.type = BillingPartyType.CURATORSHIP
    curator.display_name = "Curatelle"
    curator.billing_address = "Rue X 1\n1200 Genève"
    curator.is_active = True
    db.session.add(curator)
    db.session.flush()

    booking = Booking()
    booking.company_id = company.id
    booking.client_id = client.id
    booking.customer_name = "BOCHATAY Martine"
    booking.scheduled_time = datetime(2026, 8, 10, 10, 0, 0)
    booking.status = BookingStatus.COMPLETED.value
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.amount = 40.0
    booking.billed_to_type = "patient"
    booking.billing_party_id = curator.id
    booking.booking_type = "manual"
    db.session.add(booking)
    db.session.flush()

    bp = ensure_patient_destination_billing_party(booking)
    assert bp is not None
    assert bp.id == curator.id


def test_ensure_institution_patient_creates_patient_bp(db, company):
    inst = Institution()
    inst.public_id = str(uuid.uuid4())
    inst.name = "Clinique Test"
    inst.institution_type = "clinic"
    inst.address = "Chemin 1"
    inst.billing_address = "Chemin 1"
    db.session.add(inst)
    db.session.flush()

    patient = InstitutionPatient()
    patient.institution_id = inst.id
    patient.first_name = "Martine"
    patient.last_name = "BOCHATAY"
    patient.address = "Route d'Anières 1"
    patient.postal_code = "1247"
    patient.city = "Anières"
    db.session.add(patient)
    db.session.flush()

    clinic = _clinic_bp(db, company)
    client = _make_client(db, company)

    booking = Booking()
    booking.company_id = company.id
    booking.client_id = client.id
    booking.institution_patient_id = patient.id
    booking.customer_name = "BOCHATAY Martine"
    booking.scheduled_time = datetime(2026, 8, 10, 10, 0, 0)
    booking.status = BookingStatus.COMPLETED.value
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.amount = 40.0
    booking.billed_to_type = "patient"
    booking.billing_party_id = clinic.id
    booking.booking_type = "institution"
    db.session.add(booking)
    db.session.flush()

    bp = ensure_patient_destination_billing_party(booking)
    db.session.flush()

    assert bp is not None
    assert bp.type == BillingPartyType.PATIENT
    assert bp.external_ref == f"patient:{patient.id}"
    assert "BOCHATAY" in bp.display_name
    assert booking.billing_party_id == bp.id

    again = get_or_create_billing_party_for_institution_patient(
        company_id=company.id, institution_patient=patient
    )
    assert again.id == bp.id


def test_resolve_portfolio_skips_establishment_client_link(db, company):
    client = _make_client(db, company)
    clinic = _clinic_bp(db, company)
    from models import ClientBillingParty

    link = ClientBillingParty()
    link.client_id = client.id
    link.billing_party_id = clinic.id
    link.is_default = True
    db.session.add(link)
    db.session.flush()

    resolved = resolve_billing_party_for_portfolio_patient(
        company_id=company.id, client=client
    )
    assert resolved.type == BillingPartyType.PATIENT
    assert resolved.id != clinic.id


def test_pick_canonical_heals_establishment_bp(db, company):
    client = _make_client(db, company)
    clinic = _clinic_bp(db, company)

    booking = Booking()
    booking.company_id = company.id
    booking.client_id = client.id
    booking.customer_name = "BOCHATAY Martine"
    booking.scheduled_time = datetime(2026, 8, 10, 10, 0, 0)
    booking.status = BookingStatus.COMPLETED.value
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.amount = 40.0
    booking.billed_to_type = "patient"
    booking.billing_party_id = clinic.id
    booking.booking_type = "manual"
    db.session.add(booking)
    db.session.flush()

    canonical = pick_canonical_billing_party_id([booking])
    assert canonical is not None
    assert canonical != clinic.id
    bp = db.session.get(BillingParty, int(canonical))
    assert bp is not None
    assert bp.type == BillingPartyType.PATIENT
