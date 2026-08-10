"""Tests : annulation auto des réservations en attente de paiement client (15 min)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from models import Booking, Payment
from models.enums import BookingStatus, PaymentStatus
from services.booking.expire_unpaid_client_bookings import (
    CLIENT_ONLINE_PAYMENT_GRACE_MINUTES,
    expire_awaiting_client_payment_bookings,
)


@pytest.mark.integration
class TestExpireAwaitingClientPayment:
    def test_cancels_when_created_before_grace(
        self, db, test_company, test_client, requires_postgresql
    ):
        booking = Booking()
        booking.user_id = test_client.user_id
        booking.company_id = test_company.id
        booking.client_id = test_client.id
        booking.customer_name = f"{test_client.first_name} {test_client.last_name}"
        booking.pickup_location = "Rue A, 1200 Genève"
        booking.dropoff_location = "HUG, 1205 Genève"
        now = datetime(2001, 1, 1, tzinfo=UTC)
        booking.scheduled_time = datetime.now(UTC) + timedelta(hours=2)
        booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
        booking.amount = float(Decimal("45.00"))
        booking.billed_to_type = "patient"
        db.session.add(booking)
        db.session.flush()
        past = now - timedelta(minutes=CLIENT_ONLINE_PAYMENT_GRACE_MINUTES + 5)
        booking.created_at = past
        db.session.commit()

        # Une fenêtre temporelle dédiée évite que le test compte des paiements
        # en attente créés par d'autres fixtures dans la base PostgreSQL partagée.
        n = expire_awaiting_client_payment_bookings(
            now=now, company_id=test_company.id
        )
        assert n == 1
        db.session.refresh(booking)
        assert booking.status == BookingStatus.CANCELED
        assert booking.cancellation_reason_code == "PAYMENT_TIMEOUT"
        assert booking.cancelled_by_role == "system"

    def test_skips_when_within_grace(
        self, db, test_company, test_client, requires_postgresql
    ):
        # Une horloge isolée empêche les réservations en attente d'autres tests,
        # présentes dans PostgreSQL partagé, d'être incluses dans le décompte.
        now = datetime(2000, 1, 1, tzinfo=UTC)
        booking = Booking()
        booking.user_id = test_client.user_id
        booking.company_id = test_company.id
        booking.client_id = test_client.id
        booking.customer_name = f"{test_client.first_name} {test_client.last_name}"
        booking.pickup_location = "Rue A, 1200 Genève"
        booking.dropoff_location = "HUG, 1205 Genève"
        booking.scheduled_time = datetime.now(UTC) + timedelta(hours=2)
        booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
        booking.amount = float(Decimal("45.00"))
        booking.billed_to_type = "patient"
        db.session.add(booking)
        db.session.flush()
        booking.created_at = now - timedelta(minutes=5)
        db.session.commit()

        n = expire_awaiting_client_payment_bookings(
            now=now, company_id=test_company.id
        )
        assert n == 0
        db.session.refresh(booking)
        assert booking.status == BookingStatus.AWAITING_CLIENT_PAYMENT

    def test_skips_when_payment_completed(
        self, db, test_company, test_client, requires_postgresql
    ):
        now = datetime(2000, 1, 1, tzinfo=UTC)
        booking = Booking()
        booking.user_id = test_client.user_id
        booking.company_id = test_company.id
        booking.client_id = test_client.id
        booking.customer_name = f"{test_client.first_name} {test_client.last_name}"
        booking.pickup_location = "Rue A, 1200 Genève"
        booking.dropoff_location = "HUG, 1205 Genève"
        booking.scheduled_time = datetime.now(UTC) + timedelta(hours=2)
        booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
        booking.amount = float(Decimal("45.00"))
        booking.billed_to_type = "patient"
        db.session.add(booking)
        db.session.flush()
        booking.created_at = now - timedelta(
            minutes=CLIENT_ONLINE_PAYMENT_GRACE_MINUTES + 10
        )
        pay = Payment(
            amount=45.0,
            method="credit_card",
            status=PaymentStatus.COMPLETED,
            user_id=test_client.user_id,
            client_id=test_client.id,
            booking_id=booking.id,
        )
        db.session.add(pay)
        db.session.commit()

        n = expire_awaiting_client_payment_bookings(
            now=now, company_id=test_company.id
        )
        assert n == 0
        db.session.refresh(booking)
        assert booking.status == BookingStatus.AWAITING_CLIENT_PAYMENT

    def test_skips_when_saferpay_pending_has_token(
        self, db, test_company, test_client, requires_postgresql
    ):
        now = datetime(2000, 1, 1, tzinfo=UTC)
        booking = Booking()
        booking.user_id = test_client.user_id
        booking.company_id = test_company.id
        booking.client_id = test_client.id
        booking.customer_name = f"{test_client.first_name} {test_client.last_name}"
        booking.pickup_location = "Rue A, 1200 Genève"
        booking.dropoff_location = "HUG, 1205 Genève"
        booking.scheduled_time = datetime.now(UTC) + timedelta(hours=2)
        booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
        booking.amount = float(Decimal("45.00"))
        booking.billed_to_type = "patient"
        db.session.add(booking)
        db.session.flush()
        booking.created_at = now - timedelta(
            minutes=CLIENT_ONLINE_PAYMENT_GRACE_MINUTES + 10
        )
        pay = Payment(
            amount=45.0,
            method="credit_card",
            status=PaymentStatus.PENDING,
            user_id=test_client.user_id,
            client_id=test_client.id,
            booking_id=booking.id,
            payment_provider="saferpay",
        )
        pay.saferpay_token = "tok-session"
        db.session.add(pay)
        db.session.commit()

        n = expire_awaiting_client_payment_bookings(
            now=now, company_id=test_company.id
        )
        assert n == 0
        db.session.refresh(booking)
        assert booking.status == BookingStatus.AWAITING_CLIENT_PAYMENT

    def test_skips_when_saferpay_pending_has_transaction_id_only(
        self, db, test_company, test_client, requires_postgresql
    ):
        now = datetime(2000, 1, 1, tzinfo=UTC)
        booking = Booking()
        booking.user_id = test_client.user_id
        booking.company_id = test_company.id
        booking.client_id = test_client.id
        booking.customer_name = f"{test_client.first_name} {test_client.last_name}"
        booking.pickup_location = "Rue A, 1200 Genève"
        booking.dropoff_location = "HUG, 1205 Genève"
        booking.scheduled_time = datetime.now(UTC) + timedelta(hours=2)
        booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
        booking.amount = float(Decimal("45.00"))
        booking.billed_to_type = "patient"
        db.session.add(booking)
        db.session.flush()
        booking.created_at = now - timedelta(
            minutes=CLIENT_ONLINE_PAYMENT_GRACE_MINUTES + 10
        )
        pay = Payment(
            amount=45.0,
            method="credit_card",
            status=PaymentStatus.PENDING,
            user_id=test_client.user_id,
            client_id=test_client.id,
            booking_id=booking.id,
            payment_provider="saferpay",
        )
        pay.saferpay_token = None
        pay.saferpay_transaction_id = f"tx-orphan-{uuid.uuid4().hex[:12]}"
        db.session.add(pay)
        db.session.commit()

        n = expire_awaiting_client_payment_bookings(
            now=now, company_id=test_company.id
        )
        assert n == 0
        db.session.refresh(booking)
        assert booking.status == BookingStatus.AWAITING_CLIENT_PAYMENT
