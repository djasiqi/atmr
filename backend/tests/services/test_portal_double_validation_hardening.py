"""7B.1 — Hardening : immutabilité, races, hold, stale."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from threading import Barrier

import pytest
from sqlalchemy.exc import IntegrityError

from application.bookings.update_pending_booking import (
    UpdatePendingBookingInput,
    UpdatePendingBookingUseCase,
)
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
    OFFER_STATUS_OFFERED,
    OFFER_STATUS_STALE,
    PortalCarrierOffer,
    PortalCarrierOfferImmutabilityError,
)
from models.portal_client_transport_confirmation import (
    PortalClientTransportConfirmation,
    PortalClientTransportConfirmationImmutabilityError,
)
from models.user import User
from services.billing.portal_payment_hold import ERROR_PORTAL_CLIENT_PAYMENT_HOLD
from services.billing.portal_receivable import (
    ReceivableLineInput,
    create_portal_receivable,
)
from services.legal.confirm_portal_transport import confirm_portal_transport
from services.legal.portal_cancellation_policy import publish_cancellation_policy
from services.legal.portal_carrier_offer import create_portal_carrier_offer
from services.legal.portal_double_validation import (
    ERROR_PORTAL_OFFER_CONFLICT,
    ERROR_PORTAL_OFFER_STALE,
    FLOW_DOUBLE_VALIDATION_V2,
)
from services.legal.record_booking_contract_event import (
    record_portal_booking_created_event,
)
from services.legal.record_terms_acceptance import record_portal_terms_acceptance
from services.notifications.end_client_booking_notify import _milestone_copy


def _portal_user() -> tuple[User, Client]:
    suffix = uuid.uuid4().hex[:8]
    user = User()
    user.username = f"h1_{suffix}"
    user.email = f"h1-{suffix}@example.com"
    user.role = UserRole.client
    user.public_id = str(uuid.uuid4())
    user.phone = "+41790001122"
    user.first_name = "Hard"
    user.last_name = "En"
    user.phone_verified_at = datetime.now(UTC)
    user.set_password("password123")
    db.session.add(user)
    db.session.flush()
    client = Client()
    client.user_id = user.id
    client.company_id = None
    client.client_type = ClientType.PORTAL
    client.contact_email = user.email
    client.billing_address = "Rue H 1"
    client.domicile_address = "Rue H 1"
    client.zip_code = "1200"
    client.city = "Genève"
    db.session.add(client)
    db.session.flush()
    record_portal_terms_acceptance(user, client)
    db.session.flush()
    return user, client


def _company(*, name: str) -> tuple[Company, User]:
    suffix = uuid.uuid4().hex[:8]
    owner = User()
    owner.username = f"h1c_{suffix}"
    owner.email = f"h1c-{suffix}@example.com"
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
    publish_cancellation_policy(
        company_id=int(company.id),
        version="1.0",
        body_text="Annulation gratuite >24h. Tardive: 40%. No-show: 100%.",
    )
    db.session.flush()
    return company, owner


def _v2_booking(*, user, client, estimate=85.0, maximum=95.0) -> Booking:
    booking = Booking()
    booking.user_id = user.id
    booking.client_id = client.id
    booking.company_id = None
    booking.customer_name = f"{user.first_name} {user.last_name}"
    booking.pickup_location = "Gare Cornavin"
    booking.dropoff_location = "HUG"
    booking.scheduled_time = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=2)
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
def enable_dv(app):
    prev = app.config.get("PORTAL_DOUBLE_VALIDATION_ENABLED")
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = True
    yield
    app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = prev


def test_offer_contractual_fields_immutable(db_session, enable_dv):
    user, client = _portal_user()
    company, _ = _company(name="Imm Co")
    booking = _v2_booking(user=user, client=client)
    res = create_portal_carrier_offer(
        booking=booking, company_id=int(company.id), offered_amount=82
    )
    assert res.ok
    offer = res.offer
    offer.offered_amount = Decimal("99.00")
    with pytest.raises(PortalCarrierOfferImmutabilityError):
        db.session.flush()
    db.session.rollback()


def test_offer_hash_immutable(db_session, enable_dv):
    user, client = _portal_user()
    company, _ = _company(name="Hash Co")
    booking = _v2_booking(user=user, client=client)
    offer = create_portal_carrier_offer(
        booking=booking, company_id=int(company.id), offered_amount=82
    ).offer
    offer.offer_content_hash = "0" * 64
    with pytest.raises(PortalCarrierOfferImmutabilityError):
        db.session.flush()
    db.session.rollback()


def test_offer_delete_blocked(db_session, enable_dv):
    user, client = _portal_user()
    company, _ = _company(name="Del Co")
    booking = _v2_booking(user=user, client=client)
    res = create_portal_carrier_offer(
        booking=booking, company_id=int(company.id), offered_amount=80
    )
    assert res.ok
    db.session.delete(res.offer)
    with pytest.raises(PortalCarrierOfferImmutabilityError):
        db.session.flush()
    db.session.rollback()


def test_confirmation_references_exact_offer_and_append_only(db_session, enable_dv):
    user, client = _portal_user()
    company, _ = _company(name="Conf Co")
    booking = _v2_booking(user=user, client=client)
    offer = create_portal_carrier_offer(
        booking=booking, company_id=int(company.id), offered_amount=Decimal("82.00")
    ).offer
    conf_res = confirm_portal_transport(
        booking=booking,
        user_id=int(user.id),
        carrier_offer_id=int(offer.id),
        expected_offer_hash=offer.offer_content_hash,
    )
    assert conf_res.ok
    conf = conf_res.confirmation
    assert conf.carrier_offer_hash == offer.offer_content_hash
    assert Decimal(str(conf.contractual_amount)) == Decimal(str(offer.offered_amount))
    conf.contractual_amount = Decimal("1.00")
    with pytest.raises(PortalClientTransportConfirmationImmutabilityError):
        db.session.flush()
    db.session.rollback()


def test_maximum_snapshot_distinct_not_from_booking_amount(db_session, enable_dv):
    user, client = _portal_user()
    booking = _v2_booking(user=user, client=client, estimate=85.0, maximum=95.0)
    booking.amount = 999.0
    db.session.flush()
    event = ClientBookingContractEvent.query.filter_by(
        booking_id=booking.id, event_type=EVENT_BOOKING_CREATED
    ).one()
    assert event.estimated_amount_snapshot == 85.0
    assert event.maximum_accepted_amount_snapshot == 95.0
    assert event.amount_is_contractual is False


def test_concurrent_second_offer_rejected_sequential(db_session, enable_dv):
    user, client = _portal_user()
    company_a, _ = _company(name="Race A")
    company_b, _ = _company(name="Race B")
    booking = _v2_booking(user=user, client=client)
    first = create_portal_carrier_offer(
        booking=booking, company_id=int(company_a.id), offered_amount=82
    )
    assert first.ok
    second = create_portal_carrier_offer(
        booking=booking, company_id=int(company_b.id), offered_amount=80
    )
    assert second.ok is False
    assert second.error["error"] == ERROR_PORTAL_OFFER_CONFLICT
    assert (
        PortalCarrierOffer.query.filter_by(
            booking_id=booking.id, status=OFFER_STATUS_OFFERED
        ).count()
        == 1
    )


def test_same_company_reaccept_is_idempotent(db_session, enable_dv):
    """Re-clic Accepter par le même transporteur → succès, même offre."""
    user, client = _portal_user()
    company, _ = _company(name="Idem Co")
    booking = _v2_booking(user=user, client=client)
    first = create_portal_carrier_offer(
        booking=booking, company_id=int(company.id), offered_amount=40
    )
    assert first.ok
    second = create_portal_carrier_offer(
        booking=booking, company_id=int(company.id), offered_amount=40
    )
    assert second.ok is True
    assert second.offer is not None
    assert int(second.offer.id) == int(first.offer.id)
    assert (
        PortalCarrierOffer.query.filter_by(
            booking_id=booking.id, status=OFFER_STATUS_OFFERED
        ).count()
        == 1
    )


@pytest.mark.integration
def test_concurrent_carrier_offers_real_race(app, enable_dv):
    """Race réelle A/B (threads + Barrier) → exactement une offre offered.

    Sans fixture ``db`` (savepoint) afin que le commit soit visible cross-thread.
    """
    from sqlalchemy import text

    with app.app_context():
        user, client = _portal_user()
        company_a, _owner_a = _company(name="Thr A")
        company_b, _owner_b = _company(name="Thr B")
        booking = _v2_booking(user=user, client=client)
        booking_id = int(booking.id)
        ca_id = int(company_a.id)
        cb_id = int(company_b.id)
        db.session.commit()

    barrier = Barrier(2)
    outcomes: list[tuple[bool, str | None]] = []

    def _run(company_id: int, amount: float) -> tuple[bool, str | None]:
        barrier.wait(timeout=30)
        with app.app_context():
            app.config["PORTAL_DOUBLE_VALIDATION_ENABLED"] = True
            from models.booking import Booking as BookingModel

            b = db.session.get(BookingModel, booking_id)
            try:
                res = create_portal_carrier_offer(
                    booking=b, company_id=company_id, offered_amount=amount
                )
                if res.ok:
                    db.session.commit()
                    return True, None
                db.session.rollback()
                return False, (res.error or {}).get("error")
            except IntegrityError:
                db.session.rollback()
                return False, ERROR_PORTAL_OFFER_CONFLICT
            except Exception as exc:  # noqa: BLE001 — diagnostic course
                db.session.rollback()
                return False, type(exc).__name__

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_run, ca_id, 82.0),
                pool.submit(_run, cb_id, 80.0),
            ]
            for fut in as_completed(futures):
                outcomes.append(fut.result())

        ok_count = sum(1 for ok, _ in outcomes if ok)
        assert ok_count == 1, outcomes
        assert all(
            ok or err in (ERROR_PORTAL_OFFER_CONFLICT, "IntegrityError")
            for ok, err in outcomes
        ), outcomes

        with app.app_context():
            active = PortalCarrierOffer.query.filter_by(
                booking_id=booking_id, status=OFFER_STATUS_OFFERED
            ).count()
            assert active == 1
    finally:
        with app.app_context():
            # Offres : DELETE SQL (listeners ORM bloquent delete ORM).
            # Events contractuels : append-only DB — on ne les efface pas.
            # Commit isolé : ne pas combiner avec un UPDATE qui peut échouer
            # (sinon rollback de la suppression → pollution des autres tests).
            db.session.execute(
                text("DELETE FROM portal_carrier_offer WHERE booking_id = :b"),
                {"b": booking_id},
            )
            db.session.commit()


def test_material_modification_stales_offer(db_session, enable_dv):
    user, client = _portal_user()
    company, _ = _company(name="Stale Co")
    booking = _v2_booking(user=user, client=client)
    offer = create_portal_carrier_offer(
        booking=booking, company_id=int(company.id), offered_amount=82
    ).offer
    original_amount = Decimal(str(offer.offered_amount))
    original_hash = offer.offer_content_hash
    out = UpdatePendingBookingUseCase().execute(
        UpdatePendingBookingInput(
            booking=booking,
            validated_data={"pickup_location": "Aéroport GVA"},
        )
    )
    assert out.success
    db.session.flush()
    db.session.refresh(offer)
    assert offer.status == OFFER_STATUS_STALE
    assert Decimal(str(offer.offered_amount)) == original_amount
    assert offer.offer_content_hash == original_hash
    refused = confirm_portal_transport(
        booking=booking,
        user_id=int(user.id),
        carrier_offer_id=int(offer.id),
    )
    assert refused.ok is False
    assert refused.error["error"] == ERROR_PORTAL_OFFER_STALE
    assert (
        PortalClientTransportConfirmation.query.filter_by(booking_id=booking.id).count()
        == 0
    )
    assert booking.company_id is None


def test_payment_hold_race_blocks_confirmation(db_session, enable_dv):
    user, client = _portal_user()
    company, owner = _company(name="Hold Race")
    booking = _v2_booking(user=user, client=client)
    offer = create_portal_carrier_offer(
        booking=booking, company_id=int(company.id), offered_amount=82
    ).offer
    assert offer is not None

    done = Booking()
    done.user_id = user.id
    done.client_id = client.id
    done.company_id = company.id
    done.customer_name = booking.customer_name
    done.pickup_location = "X"
    done.dropoff_location = "Y"
    done.scheduled_time = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=5)
    done.amount = 50.0
    done.status = BookingStatus.COMPLETED
    done.is_round_trip = False
    done.portal_contract_flow = FLOW_DOUBLE_VALIDATION_V2
    db.session.add(done)
    db.session.flush()
    record_portal_booking_created_event(
        booking=done, user_id=user.id, maximum_accepted_amount=95.0
    )
    issued = datetime.now(UTC) - timedelta(days=20)
    due = datetime.now(UTC) - timedelta(days=10)
    create_portal_receivable(
        company=company,
        recorded_by_user_id=owner.id,
        external_invoice_number=f"INV-{uuid.uuid4().hex[:6]}",
        issued_at=issued,
        due_date=due,
        lines=[
            ReceivableLineInput(booking_id=done.id, invoiced_amount=Decimal("50.00"))
        ],
    )
    db.session.flush()

    refused = confirm_portal_transport(
        booking=booking,
        user_id=int(user.id),
        carrier_offer_id=int(offer.id),
    )
    assert refused.ok is False
    assert refused.error["error"] == ERROR_PORTAL_CLIENT_PAYMENT_HOLD
    assert (
        PortalClientTransportConfirmation.query.filter_by(booking_id=booking.id).count()
        == 0
    )
    db.session.refresh(booking)
    assert booking.company_id is None


def test_notification_wording_contractual(enable_dv):
    title, body = _milestone_copy(
        "carrier_offered", extra={"company_name": "X", "offered_amount": 82}
    )
    assert "proposition" in title.lower()
    assert "disponible" in title.lower()
    assert "pas encore" in body.lower()
    t2, b2 = _milestone_copy(
        "transport_confirmed",
        extra={"company_name": "X", "contractual_amount": 82},
    )
    assert "confirmé" in t2.lower()
    assert "82" in b2
