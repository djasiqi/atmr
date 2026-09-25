"""PORTAL Direct patient — opportunité facturation + montant contractuel."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal

import pytest

from application.companies.accept_reservation import AcceptReservationUseCase
from application.invoices.billable_amount import (
    SOURCE_PORTAL_CONTRACTUAL,
    calculate_billable_booking_amount,
)
from application.invoices.billing_opportunities import list_billing_opportunities
from ext import db
from models.booking import Booking
from models.client import Client
from models.company import Company
from models.enums import BookingStatus, ClientType, UserRole
from models.user import User
from services.legal.confirm_portal_transport import confirm_portal_transport
from services.legal.portal_cancellation_policy import publish_cancellation_policy
from services.legal.portal_double_validation import FLOW_DOUBLE_VALIDATION_V2
from services.legal.record_booking_contract_event import (
    record_portal_booking_created_event,
)
from services.legal.record_terms_acceptance import record_portal_terms_acceptance
from shared.time_utils import now_utc


def _portal_completed(*, estimate: float, ceiling: float, contractual: float):
    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"co_{suffix}"
    owner.email = f"co-{suffix}@example.com"
    owner.role = UserRole.company
    owner.public_id = str(uuid.uuid4())
    owner.set_password("password123")
    db.session.add(owner)
    db.session.flush()

    company = Company()
    company.name = f"Emmenez {suffix}"
    company.user_id = owner.id
    company.is_approved = True
    db.session.add(company)
    db.session.flush()

    publish_cancellation_policy(
        company_id=int(company.id),
        version="v1",
        body_text="Conditions d'annulation test.\n",
    )
    db.session.flush()

    user = User()
    user.username = f"portal_{suffix}"
    user.email = f"portal-{suffix}@example.com"
    user.role = UserRole.client
    user.first_name = "Mirjete"
    user.last_name = "Osmani"
    user.public_id = str(uuid.uuid4())
    user.phone = "+41791112233"
    user.phone_verified_at = now_utc()
    user.set_password("password123")
    db.session.add(user)
    db.session.flush()

    client = Client()
    client.user_id = user.id
    client.company_id = None
    client.client_type = ClientType.PORTAL
    client.contact_email = user.email
    client.billing_address = "Avenue Ernest-Pictet 9\n1203 Genève"
    client.domicile_address = "Avenue Ernest-Pictet 9"
    client.domicile_zip = "1203"
    client.domicile_city = "Genève"
    db.session.add(client)
    db.session.flush()
    record_portal_terms_acceptance(user, client)
    db.session.flush()

    booking = Booking()
    booking.user_id = user.id
    booking.client_id = client.id
    booking.company_id = None
    booking.customer_name = "Mme Mirjete Osmani"
    booking.pickup_location = "Avenue Ernest-Pictet 9, 1203, Genève"
    booking.dropoff_location = "HUG"
    booking.scheduled_time = datetime(2026, 9, 24, 21, 55, 0)
    booking.amount = float(estimate)
    booking.status = BookingStatus.PENDING
    booking.portal_contract_flow = FLOW_DOUBLE_VALIDATION_V2
    booking.billed_to_type = "patient"
    booking.is_round_trip = False
    db.session.add(booking)
    db.session.flush()

    record_portal_booking_created_event(
        booking=booking,
        user_id=user.id,
        maximum_accepted_amount=float(ceiling),
    )
    db.session.flush()

    accept = AcceptReservationUseCase().execute(
        booking,
        company_id=int(company.id),
        offered_amount=float(contractual),
        actor_user_id=int(owner.id),
    )
    assert accept.ok, accept.error
    assert accept.portal_offer_id is not None

    conf = confirm_portal_transport(
        booking=booking,
        user_id=int(user.id),
        carrier_offer_id=int(accept.portal_offer_id),
    )
    assert conf.ok, conf.error
    db.session.flush()

    booking.status = BookingStatus.COMPLETED
    booking.completed_at = now_utc()
    # Régression : estimation encore visible ailleurs — le montant facturable
    # doit rester le contractuel.
    booking.amount = float(estimate)
    db.session.flush()
    return company, client, booking


@pytest.fixture
def enable_dv(app):
    prev = app.config.get("PORTAL_DOUBLE_VALIDATION_ENABLED")
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = True
    yield
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = prev


@pytest.mark.usefixtures("db_session")
def test_portal_billable_amount_uses_contractual_not_estimate(enable_dv):
    _company, _client, booking = _portal_completed(
        estimate=50, ceiling=52, contractual=40
    )
    billed = calculate_billable_booking_amount(booking)
    assert billed.amount_ht == Decimal("40.00")
    assert billed.source == SOURCE_PORTAL_CONTRACTUAL
    assert float(booking.amount) == 50.0


@pytest.mark.usefixtures("db_session")
def test_portal_client_appears_in_billing_opportunities(enable_dv):
    company, client, booking = _portal_completed(
        estimate=50, ceiling=52, contractual=40
    )
    result = list_billing_opportunities(
        company_id=int(company.id),
        period_year=2026,
        period_month=9,
    )
    match = next(
        (p for p in result.patient_items if p.carrier_client_id == int(client.id)),
        None,
    )
    assert match is not None, [p.display_name for p in result.patient_items]
    assert match.can_generate is True
    assert float(match.unbilled_total_amount) == 40.0
    assert "Osmani" in (match.display_name or "") or "Mirjete" in (
        match.display_name or ""
    )
    assert int(booking.id) > 0
