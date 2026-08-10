"""Tests unitaires finalize Saferpay (mock HTTP)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from models.booking import Booking
from models.enums import BookingStatus, PaymentStatus
from models.payment import Payment
from services.saferpay.assert_response_status import (
    SAFERPAY_FINALIZE_ALREADY_COMPLETED,
    SAFERPAY_FINALIZE_ASSERT_FAILED,
    SAFERPAY_FINALIZE_ASSERT_TRANSIENT,
    SAFERPAY_FINALIZE_CAPTURE_FAILED,
    SAFERPAY_FINALIZE_COMPLETED,
    SAFERPAY_FINALIZE_PAYMENT_FAILED,
)
from services.saferpay.finalize_payment import finalize_saferpay_payment


def _tx_id(prefix: str = "tx") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


@pytest.fixture(autouse=True)
def _saferpay_env(monkeypatch):
    monkeypatch.setenv("SAFERPAY_CUSTOMER_ID", "test-customer")
    monkeypatch.setenv("SAFERPAY_TERMINAL_ID", "test-terminal")


@pytest.mark.integration
@pytest.mark.usefixtures("app_context")
def test_finalize_already_completed(
    db, sample_user, sample_client, sample_company, requires_postgresql
):
    booking = Booking()
    booking.user_id = sample_client.user_id
    booking.company_id = sample_company.id
    booking.client_id = sample_client.id
    booking.customer_name = "Test Client"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC)
    booking.status = BookingStatus.PENDING
    booking.amount = 10.0
    booking.billed_to_type = "patient"
    db.session.add(booking)
    db.session.flush()

    pay = Payment(
        amount=10.0,
        method="credit_card",
        status=PaymentStatus.COMPLETED,
        user_id=sample_user.id,
        client_id=sample_client.id,
        booking_id=booking.id,
        payment_provider="saferpay",
    )
    pay.saferpay_token = "x"
    db.session.add(pay)
    db.session.commit()

    out = finalize_saferpay_payment(pay)
    assert out["status"] == SAFERPAY_FINALIZE_ALREADY_COMPLETED


@pytest.mark.integration
@pytest.mark.usefixtures("app_context")
def test_finalize_capture_failed_persists_tx(
    db, sample_user, sample_client, sample_company, requires_postgresql
):
    booking = Booking()
    booking.user_id = sample_client.user_id
    booking.company_id = sample_company.id
    booking.client_id = sample_client.id
    booking.customer_name = "Test"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC)
    booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
    booking.amount = 10.0
    booking.billed_to_type = "patient"
    db.session.add(booking)
    db.session.flush()

    pay = Payment(
        amount=10.0,
        method="credit_card",
        status=PaymentStatus.PENDING,
        user_id=sample_user.id,
        client_id=sample_client.id,
        booking_id=booking.id,
        payment_provider="saferpay",
    )
    pay.saferpay_token = "session-token"
    db.session.add(pay)
    db.session.commit()
    pid = pay.id
    tx_id = _tx_id("capture-fail")

    def fake_post(subpath, payload):
        if "Assert" in subpath:
            return (
                200,
                {"Transaction": {"Id": tx_id, "Status": "AUTHORIZED"}},
                "{}",
            )
        if "Capture" in subpath:
            return 503, None, "unavailable"
        return 500, None, "err"

    with patch(
        "services.saferpay.finalize_payment.saferpay_post_json", side_effect=fake_post
    ):
        out = finalize_saferpay_payment(pay)

    assert out["status"] == SAFERPAY_FINALIZE_CAPTURE_FAILED
    db.session.expire_all()
    pay2 = db.session.get(Payment, pid)
    assert pay2.status == PaymentStatus.PENDING
    assert pay2.saferpay_transaction_id == tx_id


@pytest.mark.integration
@pytest.mark.usefixtures("app_context")
def test_finalize_assert_transient_no_failed(
    db, sample_user, sample_client, sample_company, requires_postgresql
):
    booking = Booking()
    booking.user_id = sample_client.user_id
    booking.company_id = sample_company.id
    booking.client_id = sample_client.id
    booking.customer_name = "Test"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC)
    booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
    booking.amount = 10.0
    booking.billed_to_type = "patient"
    db.session.add(booking)
    db.session.flush()

    pay = Payment(
        amount=10.0,
        method="credit_card",
        status=PaymentStatus.PENDING,
        user_id=sample_user.id,
        client_id=sample_client.id,
        booking_id=booking.id,
        payment_provider="saferpay",
    )
    pay.saferpay_token = "session-token"
    db.session.add(pay)
    db.session.commit()
    pid = pay.id

    with patch(
        "services.saferpay.finalize_payment.saferpay_post_json",
        return_value=(503, None, "bad"),
    ):
        out = finalize_saferpay_payment(pay)

    assert out["status"] == SAFERPAY_FINALIZE_ASSERT_TRANSIENT
    db.session.expire_all()
    pay2 = db.session.get(Payment, pid)
    assert pay2.status == PaymentStatus.PENDING


@pytest.mark.integration
@pytest.mark.usefixtures("app_context")
def test_finalize_assert_400_failed(
    db, sample_user, sample_client, sample_company, requires_postgresql
):
    booking = Booking()
    booking.user_id = sample_client.user_id
    booking.company_id = sample_company.id
    booking.client_id = sample_client.id
    booking.customer_name = "Test"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC)
    booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
    booking.amount = 10.0
    booking.billed_to_type = "patient"
    db.session.add(booking)
    db.session.flush()

    pay = Payment(
        amount=10.0,
        method="credit_card",
        status=PaymentStatus.PENDING,
        user_id=sample_user.id,
        client_id=sample_client.id,
        booking_id=booking.id,
        payment_provider="saferpay",
    )
    pay.saferpay_token = "session-token"
    db.session.add(pay)
    db.session.commit()
    pid = pay.id

    with patch(
        "services.saferpay.finalize_payment.saferpay_post_json",
        return_value=(400, None, "bad request"),
    ):
        out = finalize_saferpay_payment(pay)

    assert out["status"] == SAFERPAY_FINALIZE_ASSERT_FAILED
    db.session.expire_all()
    pay2 = db.session.get(Payment, pid)
    assert pay2.status == PaymentStatus.FAILED


@pytest.mark.integration
@pytest.mark.usefixtures("app_context")
def test_finalize_payment_failed_tx(
    db, sample_user, sample_client, sample_company, requires_postgresql
):
    booking = Booking()
    booking.user_id = sample_client.user_id
    booking.company_id = sample_company.id
    booking.client_id = sample_client.id
    booking.customer_name = "Test"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC)
    booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
    booking.amount = 10.0
    booking.billed_to_type = "patient"
    db.session.add(booking)
    db.session.flush()

    pay = Payment(
        amount=10.0,
        method="credit_card",
        status=PaymentStatus.PENDING,
        user_id=sample_user.id,
        client_id=sample_client.id,
        booking_id=booking.id,
        payment_provider="saferpay",
    )
    pay.saferpay_token = "session-token"
    db.session.add(pay)
    db.session.commit()
    failed_tx = _tx_id("failed")

    with patch(
        "services.saferpay.finalize_payment.saferpay_post_json",
        return_value=(
            200,
            {"Transaction": {"Id": failed_tx, "Status": "FAILED"}},
            "{}",
        ),
    ):
        out = finalize_saferpay_payment(pay)

    assert out["status"] == SAFERPAY_FINALIZE_PAYMENT_FAILED


@pytest.mark.integration
@pytest.mark.usefixtures("app_context")
def test_finalize_completed_happy_path(
    db, sample_user, sample_client, sample_company, requires_postgresql
):
    booking = Booking()
    booking.user_id = sample_client.user_id
    booking.company_id = sample_company.id
    booking.client_id = sample_client.id
    booking.customer_name = "Test"
    booking.pickup_location = "A"
    booking.dropoff_location = "B"
    booking.scheduled_time = datetime.now(UTC)
    booking.status = BookingStatus.AWAITING_CLIENT_PAYMENT
    booking.amount = 10.0
    booking.billed_to_type = "patient"
    db.session.add(booking)
    db.session.flush()
    bid = booking.id

    pay = Payment(
        amount=10.0,
        method="credit_card",
        status=PaymentStatus.PENDING,
        user_id=sample_user.id,
        client_id=sample_client.id,
        booking_id=booking.id,
        payment_provider="saferpay",
    )
    pay.saferpay_token = "session-token"
    db.session.add(pay)
    db.session.commit()
    ok_tx = _tx_id("ok")

    def fake_post(subpath, payload):
        if "Assert" in subpath:
            return (
                200,
                {"Transaction": {"Id": ok_tx, "Status": "AUTHORIZED"}},
                "{}",
            )
        if "Capture" in subpath:
            return 200, {"Status": "CAPTURED"}, "{}"
        return 500, None, "x"

    with patch(
        "services.saferpay.finalize_payment.saferpay_post_json", side_effect=fake_post
    ):
        out = finalize_saferpay_payment(pay)

    assert out["status"] == SAFERPAY_FINALIZE_COMPLETED
    db.session.expire_all()
    b2 = db.session.get(Booking, bid)
    assert b2.status == BookingStatus.PENDING
