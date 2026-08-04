"""BillingParty PATIENT technique pour courses portefeuille (Direct patient)."""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from application.invoices.billing_opportunities import (
    list_billing_opportunities,
    opportunities_to_dict,
)
from models import BillingParty, Booking, Client, ClientBillingParty, Company, User
from models.enums import BillingPartyType, BookingStatus, UserRole
from services.billing.billing_party_linker import (
    get_or_create_billing_party_for_direct_patient,
    resolve_billing_party_for_portfolio_patient,
)
from services.billing.direct_patient_billing_party_backfill import (
    run_backfill_direct_patient_billing_party,
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


def _make_client(
    db,
    company: Company,
    *,
    first: str = "Drin",
    last: str = "JASIQI",
    street: str = "Avenue Ernest-Pictet 9",
    zip_code: str = "1203",
    city: str = "Genève",
) -> Client:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"cli_{suffix}"
    user.email = f"cli-{suffix}@test.ch"
    user.role = UserRole.client
    user.first_name = first
    user.last_name = last
    user.public_id = str(uuid.uuid4())
    user.set_password("password123", force_change=False)
    db.session.add(user)
    db.session.flush()

    client = Client()
    client.user_id = user.id
    client.company_id = company.id
    client.domicile_address = street
    client.domicile_zip = zip_code
    client.domicile_city = city
    client.default_billed_to_type = "patient"
    db.session.add(client)
    db.session.flush()
    return client


def _make_booking(
    db,
    *,
    company: Company,
    client: Client,
    scheduled: datetime,
    billing_party_id: int | None = None,
    status: str = BookingStatus.COMPLETED.value,
) -> Booking:
    booking = Booking()
    booking.company_id = company.id
    booking.client_id = client.id
    booking.customer_name = f"{client.user.first_name} {client.user.last_name}".strip()
    booking.scheduled_time = scheduled
    booking.status = status
    booking.pickup_location = client.domicile_address or "Pickup"
    booking.dropoff_location = "HUG"
    booking.amount = 45.0
    booking.billed_to_type = "patient"
    booking.billing_party_id = billing_party_id
    booking.booking_type = "manual"
    db.session.add(booking)
    db.session.flush()
    return booking


@pytest.fixture
def company(db):
    return _make_company(db)


@pytest.fixture
def portfolio_client(db, company):
    return _make_client(db, company)


def test_get_or_create_patient_bp_creates_without_client_billing_party(
    db, company, portfolio_client
):
    bp = get_or_create_billing_party_for_direct_patient(
        company_id=company.id, client=portfolio_client
    )
    db.session.commit()

    assert bp.id is not None
    assert bp.type == BillingPartyType.PATIENT
    assert bp.external_ref == f"patient_client:{portfolio_client.id}"
    assert "Drin" in bp.display_name
    assert bp.billing_address and "1203" in bp.billing_address
    assert (
        ClientBillingParty.query.filter_by(
            client_id=portfolio_client.id, billing_party_id=bp.id
        ).count()
        == 0
    )


def test_get_or_create_patient_bp_idempotent(db, company, portfolio_client):
    bp1 = get_or_create_billing_party_for_direct_patient(
        company_id=company.id, client=portfolio_client
    )
    bp2 = get_or_create_billing_party_for_direct_patient(
        company_id=company.id, client=portfolio_client
    )
    db.session.flush()
    assert bp1.id == bp2.id
    assert (
        BillingParty.query.filter_by(
            company_id=company.id,
            external_ref=f"patient_client:{portfolio_client.id}",
        ).count()
        == 1
    )


def test_get_or_create_patient_bp_updates_snapshot(db, company, portfolio_client):
    bp = get_or_create_billing_party_for_direct_patient(
        company_id=company.id, client=portfolio_client
    )
    portfolio_client.domicile_address = "Rue Nouvelle 1"
    portfolio_client.domicile_zip = "1205"
    portfolio_client.domicile_city = "Genève"
    portfolio_client.user.first_name = "Nouveau"
    db.session.flush()

    updated = get_or_create_billing_party_for_direct_patient(
        company_id=company.id, client=portfolio_client
    )
    assert updated.id == bp.id
    assert "Nouveau" in updated.display_name
    assert "Rue Nouvelle 1" in (updated.billing_address or "")
    assert "1205" in (updated.billing_address or "")


def test_patient_bp_isolated_by_company(db, company, portfolio_client):
    other = _make_company(db, suffix=uuid.uuid4().hex[:8])
    other_client = _make_client(db, other, first="Other", last="Client")
    # Même id client impossible ; on vérifie qu'un BP d'une company
    # n'est pas réutilisé via external_ref d'une autre.
    bp_a = get_or_create_billing_party_for_direct_patient(
        company_id=company.id, client=portfolio_client
    )
    bp_b = get_or_create_billing_party_for_direct_patient(
        company_id=other.id, client=other_client
    )
    assert bp_a.id != bp_b.id
    assert bp_a.company_id == company.id
    assert bp_b.company_id == other.id


def test_resolve_portfolio_patient_prefers_third_party(
    db, company, portfolio_client
):
    curator = BillingParty()
    curator.company_id = company.id
    curator.type = BillingPartyType.CURATORSHIP
    curator.display_name = "Curatelle Test"
    curator.billing_address = "Rue Curateur 1\n1200 Genève"
    curator.is_active = True
    db.session.add(curator)
    db.session.flush()

    link = ClientBillingParty()
    link.client_id = portfolio_client.id
    link.billing_party_id = curator.id
    link.is_default = True
    db.session.add(link)
    db.session.flush()

    resolved = resolve_billing_party_for_portfolio_patient(
        company_id=company.id, client=portfolio_client
    )
    assert resolved.id == curator.id
    assert resolved.type == BillingPartyType.CURATORSHIP
    assert (
        BillingParty.query.filter_by(
            company_id=company.id,
            external_ref=f"patient_client:{portfolio_client.id}",
        ).count()
        == 0
    )


def test_resolve_portfolio_patient_creates_patient_when_no_third_party(
    db, company, portfolio_client
):
    resolved = resolve_billing_party_for_portfolio_patient(
        company_id=company.id, client=portfolio_client
    )
    assert resolved.type == BillingPartyType.PATIENT
    assert resolved.external_ref == f"patient_client:{portfolio_client.id}"


def test_opportunity_lists_patient_when_bp_present(db, company, portfolio_client):
    bp = get_or_create_billing_party_for_direct_patient(
        company_id=company.id, client=portfolio_client
    )
    _make_booking(
        db,
        company=company,
        client=portfolio_client,
        scheduled=datetime(2026, 8, 4, 0, 38),
        billing_party_id=bp.id,
    )
    db.session.commit()

    result = list_billing_opportunities(
        company_id=company.id, period_year=2026, period_month=8
    )
    payload = opportunities_to_dict(result)
    assert payload["ignored_missing_billing_party_count"] == 0
    assert len(result.patient_items) == 1
    item = result.patient_items[0]
    assert item.billing_party_id == bp.id
    assert item.can_generate is True
    assert item.carrier_client_id == portfolio_client.id


def test_opportunity_counts_ignored_missing_bp(db, company, portfolio_client):
    _make_booking(
        db,
        company=company,
        client=portfolio_client,
        scheduled=datetime(2026, 8, 4, 0, 38),
        billing_party_id=None,
    )
    db.session.commit()

    result = list_billing_opportunities(
        company_id=company.id, period_year=2026, period_month=8
    )
    payload = opportunities_to_dict(result)
    assert payload["ignored_missing_billing_party_count"] == 1
    assert result.patient_items == []


def test_backfill_assigns_patient_bp_idempotent(db, company, portfolio_client):
    booking = _make_booking(
        db,
        company=company,
        client=portfolio_client,
        scheduled=datetime(2026, 8, 4, 0, 38),
        billing_party_id=None,
    )
    db.session.commit()

    first = run_backfill_direct_patient_billing_party(
        dry_run=False, company_id=company.id
    )
    assert first.bookings_updated == 1
    assert first.clients_touched == 1
    db.session.refresh(booking)
    assert booking.billing_party_id is not None
    bp_id = booking.billing_party_id

    second = run_backfill_direct_patient_billing_party(
        dry_run=False, company_id=company.id
    )
    assert second.bookings_updated == 0
    assert (
        BillingParty.query.filter_by(
            company_id=company.id,
            external_ref=f"patient_client:{portfolio_client.id}",
        ).count()
        == 1
    )
    db.session.refresh(booking)
    assert booking.billing_party_id == bp_id


def test_backfill_prefers_third_party(db, company, portfolio_client):
    curator = BillingParty()
    curator.company_id = company.id
    curator.type = BillingPartyType.CURATORSHIP
    curator.display_name = "Curateur"
    curator.billing_address = "Adresse Curateur 1, 1200 Genève"
    curator.is_active = True
    db.session.add(curator)
    db.session.flush()
    link = ClientBillingParty()
    link.client_id = portfolio_client.id
    link.billing_party_id = curator.id
    link.is_default = True
    db.session.add(link)

    booking = _make_booking(
        db,
        company=company,
        client=portfolio_client,
        scheduled=datetime(2026, 8, 4, 0, 38),
        billing_party_id=None,
    )
    db.session.commit()

    run_backfill_direct_patient_billing_party(dry_run=False, company_id=company.id)
    db.session.refresh(booking)
    assert booking.billing_party_id == curator.id
