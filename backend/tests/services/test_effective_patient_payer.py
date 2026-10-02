"""Un BP PATIENT technique ne masque pas une curatelle par défaut."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal

import pytest

from application.invoices.force_regenerate_invoice_pdf import (
    reconcile_draft_invoice_billing_party,
)
from application.invoices.invoice_candidates import list_patient_invoice_candidates
from models import (
    BillingParty,
    Booking,
    Client,
    ClientBillingParty,
    Company,
    Invoice,
    InvoiceLine,
    User,
)
from models.enums import (
    BillingPartyType,
    BookingStatus,
    InvoiceBillingStrategy,
    InvoiceLineType,
    InvoiceStatus,
    UserRole,
)
from services.billing.effective_patient_payer import (
    resolve_effective_party_id_for_bookings,
    resolve_effective_patient_billing_party,
)
from services.documents.invoice_recipient import resolve_invoice_billed_to


def _company(db) -> Company:
    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"co_{suffix}"
    owner.email = f"co-{suffix}@test.ch"
    owner.role = UserRole.company
    owner.public_id = str(uuid.uuid4())
    owner.set_password("password123", force_change=False)
    db.session.add(owner)
    db.session.flush()
    company = Company()
    company.name = f"Transport {suffix}"
    company.user_id = owner.id
    db.session.add(company)
    db.session.flush()
    return company


def _client(db, company: Company) -> Client:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"cli_{suffix}"
    user.email = f"cli-{suffix}@test.ch"
    user.role = UserRole.client
    user.first_name = "Monica Suzanne"
    user.last_name = "BRENNER"
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()
    client = Client()
    client.user_id = user.id
    client.company_id = company.id
    client.domicile_address = "Rue Ferrier 7A"
    client.domicile_zip = "1202"
    client.domicile_city = "Genève"
    client.default_billed_to_type = "patient"
    db.session.add(client)
    db.session.flush()
    return client


def _party(
    db, company, *, party_type, name, address, external_ref=None
) -> BillingParty:
    party = BillingParty()
    party.company_id = company.id
    party.type = party_type
    party.display_name = name
    party.billing_address = address
    party.is_active = True
    party.external_ref = external_ref
    db.session.add(party)
    db.session.flush()
    return party


def _link(db, client, party, *, contact_name="Jérôme GASSER", is_default=True):
    link = ClientBillingParty()
    link.client_id = client.id
    link.billing_party_id = party.id
    link.contact_name = contact_name
    link.is_default = is_default
    db.session.add(link)
    db.session.flush()
    return link


def _booking(db, company, client, party) -> Booking:
    booking = Booking()
    booking.company_id = company.id
    booking.client_id = client.id
    booking.customer_name = "Monica Suzanne BRENNER"
    booking.scheduled_time = datetime(2026, 9, 15, 10, 0, 0)
    booking.status = BookingStatus.COMPLETED.value
    booking.pickup_location = "Rue Ferrier 7A"
    booking.dropoff_location = "HUG"
    booking.amount = 40.0
    booking.billed_to_type = "patient"
    booking.billing_party_id = party.id
    booking.booking_type = "manual"
    db.session.add(booking)
    db.session.flush()
    return booking


def _technical_patient(db, company, client) -> BillingParty:
    return _party(
        db,
        company,
        party_type=BillingPartyType.PATIENT,
        name="Monica Suzanne BRENNER",
        address="Rue Ferrier 7A\n1202 Genève",
        external_ref=f"patient_client:{client.id}",
    )


def _opad(db, company) -> BillingParty:
    return _party(
        db,
        company,
        party_type=BillingPartyType.CURATORSHIP,
        name="OPAD (Office de Protection de l'Adulte)",
        address="Rte des Jeunes 1c, 1227 Genève, Suisse",
    )


@pytest.fixture
def world(db):
    company = _company(db)
    client = _client(db, company)
    patient_bp = _technical_patient(db, company, client)
    opad = _opad(db, company)
    _link(db, client, opad)
    booking = _booking(db, company, client, patient_bp)
    return company, client, patient_bp, opad, booking


def test_technical_patient_does_not_shadow_default_curatorship(db, world):
    company, _client, patient_bp, opad, booking = world
    assert booking.billing_party_id == patient_bp.id
    resolved = resolve_effective_patient_billing_party(
        booking=booking, company_id=company.id
    )
    assert resolved is not None
    assert resolved.id == opad.id
    assert booking.billing_party_id == patient_bp.id


def test_candidate_uses_opad_without_writing_booking(db, world):
    company, _client, patient_bp, opad, booking = world
    stored = booking.billing_party_id
    payload = list_patient_invoice_candidates(
        company_id=company.id, period_year=2026, period_month=9
    )
    assert payload["patients"]
    assert payload["patients"][0]["billing_party_id"] == opad.id
    assert "billing_party:" + str(opad.id) in payload["patients"][0]["id"]
    db.session.refresh(booking)
    assert booking.billing_party_id == stored == patient_bp.id
    assert booking not in db.session.dirty


def test_generate_authority_resolves_stale_patient_key(db, world):
    """La clé d'opportunité ne fait pas foi : le serveur re-résout vers OPAD."""
    company, _client, patient_bp, opad, booking = world
    assert booking.billing_party_id == patient_bp.id
    effective = resolve_effective_party_id_for_bookings(
        [booking], company_id=company.id
    )
    assert effective == opad.id


def test_pdf_renders_care_of_opad(db, world):
    company, client, _patient_bp, opad, _booking = world
    invoice = Invoice(
        company=company,
        client=client,
        invoice_number=f"INV-{uuid.uuid4().hex[:6]}",
        period_year=2026,
        period_month=9,
        status=InvoiceStatus.DRAFT,
        issued_at=datetime.now(UTC),
        due_date=datetime.now(UTC),
        subtotal_amount=Decimal("40.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("40.00"),
        billing_party_id=opad.id,
    )
    invoice.billing_strategy = InvoiceBillingStrategy.S1_PATIENT
    db.session.add(invoice)
    db.session.flush()
    resolved = resolve_invoice_billed_to(invoice)
    assert resolved.mode == "patient_care_of"
    assert resolved.addressee == "Monica Suzanne BRENNER"
    assert resolved.care_of == "OPAD (Office de Protection de l'Adulte)"
    assert resolved.attention == "Jérôme GASSER"
    assert "Rte des Jeunes 1c" in (resolved.address or "")
    assert "Rue Ferrier" not in (resolved.address or "")


def test_patient_without_third_party_stays_self(db):
    company = _company(db)
    client = _client(db, company)
    patient_bp = _technical_patient(db, company, client)
    booking = _booking(db, company, client, patient_bp)
    resolved = resolve_effective_patient_billing_party(
        booking=booking, company_id=company.id
    )
    assert resolved is not None
    assert resolved.id == patient_bp.id
    invoice = Invoice(
        company=company,
        client=client,
        invoice_number=f"INV-{uuid.uuid4().hex[:6]}",
        period_year=2026,
        period_month=9,
        status=InvoiceStatus.DRAFT,
        issued_at=datetime.now(UTC),
        due_date=datetime.now(UTC),
        subtotal_amount=Decimal("40.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("40.00"),
        billing_party_id=patient_bp.id,
    )
    invoice.billing_strategy = InvoiceBillingStrategy.S1_PATIENT
    db.session.add(invoice)
    db.session.flush()
    assert resolve_invoice_billed_to(invoice).mode == "patient_self"


def test_explicit_curatorship_is_kept(db, world):
    company, client, _patient_bp, opad, booking = world
    other = _party(
        db,
        company,
        party_type=BillingPartyType.FAMILY,
        name="Famille BRENNER",
        address="Rue de la Famille 1, 1200 Genève",
    )
    booking.billing_party_id = other.id
    db.session.flush()
    resolved = resolve_effective_patient_billing_party(
        booking=booking, company_id=company.id
    )
    assert resolved is not None
    assert resolved.id == other.id
    assert resolved.id != opad.id
    _ = client


def test_locked_patient_override_is_not_replaced(db, world):
    company, _client, patient_bp, _opad, booking = world
    booking.billing_locked_at = datetime.now(UTC)
    db.session.flush()
    resolved = resolve_effective_patient_billing_party(
        booking=booking, company_id=company.id
    )
    assert resolved is not None
    assert resolved.id == patient_bp.id


def test_voucher_keeps_priority(monkeypatch, db, world):
    company, _client, _patient_bp, _opad, booking = world
    clinic = _party(
        db,
        company,
        party_type=BillingPartyType.CLINIC,
        name="Clinique Test",
        address="Rue Clinique 1",
    )

    monkeypatch.setattr(
        "services.billing.client_stay_resolver.find_valid_voucher_for_booking",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "services.billing.client_stay_resolver.resolve_payer_from_voucher",
        lambda **_kwargs: {"billing_party_id": clinic.id},
    )
    resolved = resolve_effective_patient_billing_party(
        booking=booking, company_id=company.id
    )
    assert resolved is not None
    assert resolved.id == clinic.id


def test_active_stay_keeps_priority(monkeypatch, db, world):
    company, _client, _patient_bp, _opad, booking = world
    clinic = _party(
        db,
        company,
        party_type=BillingPartyType.CLINIC,
        name="Clinique Séjour",
        address="Rue Séjour 2",
    )
    monkeypatch.setattr(
        "services.billing.client_stay_resolver.find_valid_voucher_for_booking",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        "services.billing.client_stay_resolver.find_active_stay_for_booking",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        "services.billing.client_stay_resolver.resolve_payer_from_stay",
        lambda **_kwargs: {"billing_party_id": clinic.id},
    )
    resolved = resolve_effective_patient_billing_party(
        booking=booking, company_id=company.id
    )
    assert resolved is not None
    assert resolved.id == clinic.id


def _draft(db, company, client, party) -> tuple[Invoice, InvoiceLine]:
    invoice = Invoice(
        company=company,
        client=client,
        invoice_number=f"INV-{uuid.uuid4().hex[:6]}",
        period_year=2026,
        period_month=9,
        status=InvoiceStatus.DRAFT,
        issued_at=datetime.now(UTC),
        due_date=datetime.now(UTC),
        subtotal_amount=Decimal("40.00"),
        vat_total_amount=Decimal("0.00"),
        total_amount=Decimal("40.00"),
        billing_party_id=party.id,
    )
    invoice.billing_strategy = InvoiceBillingStrategy.S1_PATIENT
    invoice.meta = {
        "billed_to_snapshot": {"mode": "patient_self", "addressee": "Monica"},
        "recipient_snapshot": {"billing_party_id": party.id, "type": "patient"},
    }
    db.session.add(invoice)
    db.session.flush()
    line = InvoiceLine(
        invoice=invoice,
        type=InvoiceLineType.RIDE,
        description="Transport",
        qty=Decimal("1.00"),
        unit_price=Decimal("40.00"),
        line_total=Decimal("40.00"),
        vat_rate=Decimal("0.00"),
        vat_amount=Decimal("0.00"),
        total_with_vat=Decimal("40.00"),
    )
    db.session.add(line)
    db.session.flush()
    return invoice, line


def test_draft_reconciles_to_opad(db, world):
    company, client, patient_bp, opad, booking = world
    invoice, line = _draft(db, company, client, patient_bp)
    booking.invoice_line_id = line.id
    db.session.flush()
    assert reconcile_draft_invoice_billing_party(invoice) is True
    assert invoice.billing_party_id == opad.id
    assert "billed_to_snapshot" not in (invoice.meta or {})
    assert invoice.total_amount == Decimal("40.00")


def test_sent_invoice_snapshot_is_not_mutated(db, world):
    company, client, patient_bp, _opad, booking = world
    invoice, line = _draft(db, company, client, patient_bp)
    booking.invoice_line_id = line.id
    invoice.status = InvoiceStatus.SENT
    db.session.flush()
    assert reconcile_draft_invoice_billing_party(invoice) is False
    assert invoice.billing_party_id == patient_bp.id
    assert invoice.meta["billed_to_snapshot"]["mode"] == "patient_self"
