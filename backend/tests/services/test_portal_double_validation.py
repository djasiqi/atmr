"""7B — Double validation PORTAL (plafond, offre, confirmation)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from application.companies.accept_reservation import AcceptReservationUseCase
from ext import db
from models.booking import Booking
from models.client import Client
from models.client_booking_contract_event import (
    EVENT_BOOKING_CREATED,
    ClientBookingContractEvent,
)
from models.company import Company
from models.enums import BookingStatus, ClientType, UserRole
from models.portal_carrier_offer import (
    OFFER_STATUS_CONFIRMED,
    OFFER_STATUS_OFFERED,
    PortalCarrierOffer,
)
from models.portal_client_transport_confirmation import (
    PortalClientTransportConfirmation,
)
from models.user import User
from services.legal.confirm_portal_transport import confirm_portal_transport
from services.legal.portal_cancellation_policy import publish_cancellation_policy
from services.legal.portal_carrier_offer import create_portal_carrier_offer
from services.legal.portal_double_validation import (
    ERROR_PORTAL_OFFER_ABOVE_CLIENT_LIMIT,
    ERROR_PORTAL_OFFER_CONFLICT,
    FLOW_DOUBLE_VALIDATION_V2,
    is_portal_double_validation_enabled,
)
from services.legal.record_booking_contract_event import (
    record_portal_booking_created_event,
)
from services.legal.record_terms_acceptance import record_portal_terms_acceptance


def _portal_user() -> tuple[User, Client]:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"dv_{suffix}"
    user.email = f"dv-{suffix}@example.com"
    user.role = UserRole.client
    user.public_id = str(uuid.uuid4())
    user.phone = "+41791112233"
    user.first_name = "Alice"
    user.last_name = "Martin"
    user.phone_verified_at = datetime.now(UTC)
    user.set_password("password123")
    db.session.add(user)
    db.session.flush()
    client = Client()
    client.user_id = user.id
    client.company_id = None
    client.client_type = ClientType.PORTAL
    client.contact_email = user.email
    client.billing_address = "Rue DV 1"
    db.session.add(client)
    db.session.flush()
    record_portal_terms_acceptance(user, client)
    db.session.flush()
    return user, client


def _company(*, name: str) -> Company:
    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"dvc_{suffix}"
    owner.email = f"dvc-{suffix}@example.com"
    owner.role = UserRole.company
    owner.public_id = str(uuid.uuid4())
    owner.set_password("password123")
    db.session.add(owner)
    db.session.flush()
    company = Company()
    company.name = name
    company.user_id = owner.id
    company.is_approved = True
    db.session.add(company)
    db.session.flush()
    return company


def _v2_booking(*, user: User, client: Client, estimate: float, maximum: float) -> Booking:
    booking = Booking()
    booking.user_id = user.id
    booking.client_id = client.id
    booking.company_id = None
    booking.customer_name = f"{user.first_name} {user.last_name}"
    booking.pickup_location = "Genève Gare"
    booking.dropoff_location = "HUG"
    booking.scheduled_time = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1)
    booking.amount = float(estimate)
    booking.status = BookingStatus.PENDING
    booking.portal_contract_flow = FLOW_DOUBLE_VALIDATION_V2
    booking.is_round_trip = False
    db.session.add(booking)
    db.session.flush()
    record_portal_booking_created_event(
        booking=booking,
        user_id=user.id,
        maximum_accepted_amount=float(maximum),
    )
    db.session.flush()
    return booking


@pytest.fixture
def enable_double_validation(app):
    previous = app.config.get("PORTAL_DOUBLE_VALIDATION_ENABLED")
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = True
    yield
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = previous


def test_feature_flag_off_by_default(app):
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = False
    with app.app_context():
        assert is_portal_double_validation_enabled() is False


def test_estimate_and_maximum_stored_distinctly(db_session, enable_double_validation):
    user, client = _portal_user()
    booking = _v2_booking(user=user, client=client, estimate=85.0, maximum=95.0)
    event = ClientBookingContractEvent.query.filter_by(
        booking_id=booking.id, event_type=EVENT_BOOKING_CREATED
    ).one()
    assert event.estimated_amount_snapshot == 85.0
    assert event.maximum_accepted_amount_snapshot == 95.0
    assert event.amount_is_contractual is False
    assert event.estimated_amount_snapshot != event.maximum_accepted_amount_snapshot


def test_offer_price_gate_and_no_assignment(db_session, enable_double_validation):
    user, client = _portal_user()
    company = _company(name="Transport DV")
    publish_cancellation_policy(
        company_id=int(company.id),
        version="1.0",
        body_text="Annulation gratuite jusqu'à 24h. No-show: 50%.",
    )
    db.session.flush()
    booking = _v2_booking(user=user, client=client, estimate=85.0, maximum=95.0)

    rejected = create_portal_carrier_offer(
        booking=booking, company_id=int(company.id), offered_amount=96
    )
    assert rejected.ok is False
    assert rejected.error["error"] == ERROR_PORTAL_OFFER_ABOVE_CLIENT_LIMIT
    assert "95" not in str(rejected.error)

    ok_edge = create_portal_carrier_offer(
        booking=booking, company_id=int(company.id), offered_amount=95
    )
    assert ok_edge.ok is True
    assert booking.company_id is None
    assert (
        PortalClientTransportConfirmation.query.filter_by(
            booking_id=booking.id
        ).count()
        == 0
    )

    # Même transporteur : rejeu idempotent (offre inchangée, montant 95).
    same_carrier = create_portal_carrier_offer(
        booking=booking, company_id=int(company.id), offered_amount=82
    )
    assert same_carrier.ok is True
    assert same_carrier.offer is not None
    assert same_carrier.offer.id == ok_edge.offer.id
    assert float(same_carrier.offer.offered_amount) == 95.0

    # Autre transporteur : conflit (une seule offre active).
    other = _company(name="Transport DV Autre")
    publish_cancellation_policy(
        company_id=int(other.id),
        version="1.0",
        body_text="Annulation: voir barème entreprise.",
    )
    db.session.flush()
    conflict = create_portal_carrier_offer(
        booking=booking, company_id=int(other.id), offered_amount=82
    )
    assert conflict.ok is False
    assert conflict.error["error"] == ERROR_PORTAL_OFFER_CONFLICT


def test_accept_reservation_portal_v2_creates_offer_not_contract(
    db_session, enable_double_validation
):
    user, client = _portal_user()
    company = _company(name="Transport Offer")
    publish_cancellation_policy(
        company_id=int(company.id),
        version="1.0",
        body_text="Annulation: voir barème entreprise.",
    )
    db.session.flush()
    booking = _v2_booking(user=user, client=client, estimate=85.0, maximum=95.0)

    result = AcceptReservationUseCase().execute(
        booking, company_id=int(company.id), offered_amount=82
    )
    assert result.ok is True
    assert result.portal_offer_pending_client is True
    assert booking.company_id is None
    assert str(getattr(booking.status, "value", booking.status)).upper() == "PENDING"
    offer = PortalCarrierOffer.query.filter_by(
        booking_id=booking.id, status=OFFER_STATUS_OFFERED
    ).one()
    assert float(offer.offered_amount) == 82.0


def test_second_click_forms_contract_idempotent(db_session, enable_double_validation):
    user, client = _portal_user()
    company = _company(name="Transport Confirm")
    publish_cancellation_policy(
        company_id=int(company.id),
        version="1.0",
        body_text="No-show CHF 60. Attente: selon tarif X.",
    )
    db.session.flush()
    booking = _v2_booking(user=user, client=client, estimate=85.0, maximum=95.0)
    offer_result = create_portal_carrier_offer(
        booking=booking, company_id=int(company.id), offered_amount=Decimal("82.00")
    )
    assert offer_result.ok
    offer = offer_result.offer

    first = confirm_portal_transport(
        booking=booking,
        user_id=int(user.id),
        carrier_offer_id=int(offer.id),
        expected_offer_hash=offer.offer_content_hash,
    )
    assert first.ok is True
    db.session.refresh(booking)
    db.session.refresh(offer)
    assert booking.company_id == company.id
    assert float(booking.amount) == 82.0
    conf = PortalClientTransportConfirmation.query.filter_by(
        booking_id=booking.id
    ).one()
    assert float(conf.contractual_amount) == 82.0
    assert conf.carrier_offer_hash == offer.offer_content_hash
    assert offer.status == OFFER_STATUS_CONFIRMED

    second = confirm_portal_transport(
        booking=booking,
        user_id=int(user.id),
        carrier_offer_id=int(offer.id),
    )
    assert second.ok is True
    assert (
        PortalClientTransportConfirmation.query.filter_by(booking_id=booking.id).count()
        == 1
    )


def test_legacy_booking_no_fake_maximum(db_session):
    user, client = _portal_user()
    booking = Booking()
    booking.user_id = user.id
    booking.client_id = client.id
    booking.company_id = None
    booking.customer_name = "Legacy"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=1)
    booking.amount = 50.0
    booking.status = BookingStatus.PENDING
    booking.portal_contract_flow = None
    booking.is_round_trip = False
    db.session.add(booking)
    db.session.flush()
    event = record_portal_booking_created_event(booking=booking, user_id=user.id)
    assert event.maximum_accepted_amount_snapshot is None
    assert (
        PortalCarrierOffer.query.filter_by(booking_id=booking.id).count() == 0
    )
    assert (
        PortalClientTransportConfirmation.query.filter_by(booking_id=booking.id).count()
        == 0
    )
