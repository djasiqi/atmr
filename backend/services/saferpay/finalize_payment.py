"""Assert + Capture Saferpay après retour client ou notification."""

from __future__ import annotations

import logging
import os
import uuid
from http import HTTPStatus
from typing import Any

from ext import db
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
    SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS,
)
from services.saferpay.config import saferpay_spec_version
from services.saferpay.http_client import saferpay_post_json

logger = logging.getLogger(__name__)


def _saferpay_locking_disabled() -> bool:
    raw = (os.getenv("SAFERPAY_DISABLE_TRANSACTION_LOCKS") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _supports_row_locking() -> bool:
    if _saferpay_locking_disabled():
        return False
    bind = db.session.get_bind()
    return bind.dialect.name != "sqlite"


def _get_payment_for_finalize(payment_id: int) -> Payment | None:
    query = Payment.query.filter(Payment.id == payment_id)
    if _supports_row_locking():
        query = query.with_for_update()
    return query.first()


def _request_header(customer_id: str) -> dict[str, Any]:
    return {
        "SpecVersion": saferpay_spec_version(),
        "CustomerId": customer_id,
        "RequestId": uuid.uuid4().hex,
        "RetryIndicator": 0,
    }


def _promote_booking_if_awaiting(booking_id: int) -> None:
    booking_row = db.session.get(Booking, booking_id)
    if (
        booking_row is not None
        and booking_row.status == BookingStatus.AWAITING_CLIENT_PAYMENT
    ):
        booking_row.update_status(BookingStatus.PENDING)


def _assert_http_is_transient(st_code: int) -> bool:
    """Erreur réseau / surcharge / timeout côté transport — pas d'échec métier terminal."""
    return (
        st_code == 0
        or st_code >= HTTPStatus.INTERNAL_SERVER_ERROR
        or st_code
        in (HTTPStatus.REQUEST_TIMEOUT, HTTPStatus.TOO_MANY_REQUESTS)  # 408, 429
    )


def finalize_saferpay_payment(payment_row: Payment) -> dict[str, Any]:
    """PaymentPage Assert puis Transaction Capture si AUTHORIZED. Idempotent si déjà COMPLETED."""
    locked_payment = _get_payment_for_finalize(int(payment_row.id))
    if locked_payment is None:
        return {
            "status": SAFERPAY_FINALIZE_ASSERT_FAILED,
            "payment_id": int(payment_row.id),
        }
    payment_row = locked_payment

    if payment_row.status == PaymentStatus.COMPLETED:
        return {
            "status": SAFERPAY_FINALIZE_ALREADY_COMPLETED,
            "payment_id": payment_row.id,
        }

    token = (payment_row.saferpay_token or "").strip()
    if not token:
        raise ValueError("Paiement Saferpay sans token de session")

    customer_id = os.environ["SAFERPAY_CUSTOMER_ID"].strip()
    assert_body: dict[str, Any] = {
        "RequestHeader": _request_header(customer_id),
        "Token": token,
    }
    st_code, assert_data, raw = saferpay_post_json(
        "Payment/v1/PaymentPage/Assert",
        assert_body,
    )
    if st_code != HTTPStatus.OK or not assert_data:
        if _assert_http_is_transient(st_code):
            logger.warning(
                "Saferpay Assert transitoire payment_id=%s http=%s",
                payment_row.id,
                st_code,
            )
            return {
                "status": SAFERPAY_FINALIZE_ASSERT_TRANSIENT,
                "payment_id": payment_row.id,
                "http_status": st_code,
                "detail": (raw or "")[:500] if raw else None,
            }
        payment_row.status = PaymentStatus.FAILED
        db.session.commit()
        detail_src = (
            raw if raw else (str(assert_data) if assert_data is not None else "")
        )
        return {
            "status": SAFERPAY_FINALIZE_ASSERT_FAILED,
            "payment_id": payment_row.id,
            "http_status": st_code,
            "detail": detail_src[:500] if detail_src else None,
        }

    tx = assert_data.get("Transaction") or {}
    tx_id = (tx.get("Id") or "").strip()
    tx_status = (tx.get("Status") or "").strip().upper()
    if tx_id:
        payment_row.saferpay_transaction_id = tx_id

    if tx_status in {"FAILED", "CANCELED", "VOIDED"}:
        payment_row.status = PaymentStatus.FAILED
        db.session.commit()
        return {
            "status": SAFERPAY_FINALIZE_PAYMENT_FAILED,
            "payment_id": payment_row.id,
            "tx_status": tx_status,
        }

    if tx_status == "AUTHORIZED":
        db.session.flush()
        cap_body = {
            "RequestHeader": _request_header(customer_id),
            "TransactionReference": {"TransactionId": tx_id},
        }
        cap_st, cap_data, cap_raw = saferpay_post_json(
            "Payment/v1/Transaction/Capture",
            cap_body,
        )
        if cap_st != HTTPStatus.OK or not cap_data:
            logger.error(
                "Saferpay Capture échoué payment_id=%s tx=%s: %s",
                payment_row.id,
                tx_id,
                (cap_raw or "")[:500],
            )
            db.session.commit()
            return {
                "status": SAFERPAY_FINALIZE_CAPTURE_FAILED,
                "payment_id": payment_row.id,
                "http_status": cap_st,
                "detail": (cap_raw or "")[:500] if cap_raw else None,
            }

    elif tx_status == "CAPTURED":
        pass
    else:
        logger.warning(
            "Saferpay Assert statut inattendu payment_id=%s status=%s",
            payment_row.id,
            tx_status,
        )
        payment_row.status = PaymentStatus.FAILED
        db.session.commit()
        return {
            "status": SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS,
            "payment_id": payment_row.id,
            "tx_status": tx_status,
        }

    payment_row.status = PaymentStatus.COMPLETED
    booking_id_val = getattr(payment_row, "booking_id", None)
    if booking_id_val is not None:
        _promote_booking_if_awaiting(int(booking_id_val))
    db.session.commit()
    logger.info(
        "Saferpay paiement finalisé payment_id=%s booking_id=%s",
        payment_row.id,
        payment_row.booking_id,
    )
    if booking_id_val is not None:
        from services.dispatch.open_booking_offers import (
            refresh_dispatch_offers_after_online_payment,
        )

        refresh_dispatch_offers_after_online_payment(int(booking_id_val))
    return {"status": SAFERPAY_FINALIZE_COMPLETED, "payment_id": payment_row.id}
