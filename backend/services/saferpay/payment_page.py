"""Payment Page Initialize — redirection navigateur vers Saferpay."""

from __future__ import annotations

import logging
import os
import secrets
import uuid
from decimal import Decimal
from http import HTTPStatus
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from ext import db
from models.enums import BookingStatus, PaymentStatus
from models.payment import Payment
from services.saferpay.config import (
    saferpay_api_base_url,
    saferpay_configured,
    saferpay_public_api_prefix,
    saferpay_spec_version,
)
from services.saferpay.finalize_payment import finalize_saferpay_payment
from services.saferpay.http_client import saferpay_post_json
from services.saferpay.money import chf_amount_to_saferpay_value_str
from services.saferpay.return_urls import (
    default_saferpay_return_urls,
    validate_return_url_override,
)

if TYPE_CHECKING:
    from domain.client_dto import ClientDTO
    from models.booking import Booking
    from models.client import Client
    from models.user import User


def _decimal_amount(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value or 0))


def _client_booking_payable_amount_chf(booking: Any) -> Decimal:
    """Montant à encaisser pour le dossier : aller + retour lié si présent."""
    base = _decimal_amount(getattr(booking, "amount", 0))
    if bool(getattr(booking, "is_return", False)):
        return base
    ret = getattr(booking, "return_trip", None)
    if ret is not None:
        return base + _decimal_amount(getattr(ret, "amount", 0))
    return base

logger = logging.getLogger(__name__)

_PAYABLE_CLIENT_STATUSES = frozenset(
    {BookingStatus.PENDING, BookingStatus.AWAITING_CLIENT_PAYMENT}
)

# Saferpay Amount.Value : plus petite unite (centimes) pour CHF.
_SAFERPAY_MIN_AMOUNT_CENTS = 50


def _saferpay_locking_disabled() -> bool:
    raw = (os.getenv("SAFERPAY_DISABLE_TRANSACTION_LOCKS") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _supports_row_locking() -> bool:
    if _saferpay_locking_disabled():
        return False
    bind = db.session.get_bind()
    return bind.dialect.name != "sqlite"


def _lock_booking_row(booking_id: int) -> None:
    # Serialize initialize requests per booking to avoid duplicate active payments.
    from models.booking import Booking

    query = Booking.query.filter(Booking.id == booking_id)
    if _supports_row_locking():
        query = query.with_for_update()
    query.first()


def _return_urls_with_outcome_override(
    base: str,
    *,
    booking_id: int,
    payment_id: int,
) -> tuple[str, str, str]:
    """Success / Fail / Abort avec ``outcome=`` + corrélation (comme les URLs par défaut)."""
    parsed = urlparse(base.strip())
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    if "bookingId" not in q and "booking_id" not in q:
        q["bookingId"] = str(booking_id)
    if "paymentId" not in q and "payment_id" not in q:
        q["paymentId"] = str(payment_id)

    def _build(outcome: str) -> str:
        merged = {**q, "outcome": outcome}
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                urlencode(merged),
                parsed.fragment,
            )
        )

    return _build("success"), _build("fail"), _build("abort")


def _request_header(customer_id: str) -> dict[str, Any]:
    return {
        "SpecVersion": saferpay_spec_version(),
        "CustomerId": customer_id,
        "RequestId": uuid.uuid4().hex,
        "RetryIndicator": 0,
    }


def create_saferpay_payment_page_initialize(
    *,
    booking: Booking,
    user: User,
    client: Client | ClientDTO,
    return_url_override: str | None = None,
) -> dict[str, Any]:
    if not saferpay_configured():
        raise RuntimeError("Saferpay n'est pas configuré sur ce serveur")

    if booking.status not in _PAYABLE_CLIENT_STATUSES:
        raise ValueError(
            "Seules les reservations en attente de paiement ou de confirmation "
            + "peuvent etre payees en ligne"
        )
    booking_client_id: int | None = getattr(booking, "client_id", None)
    if booking_client_id is None or booking_client_id != client.id:
        raise ValueError("Cette réservation n'appartient pas à ce client")

    if getattr(booking, "company_id", None) is None:
        from services.dispatch.open_booking_offers import (
            seed_dispatch_offers_for_unassigned_booking,
        )

        try:
            seed_dispatch_offers_for_unassigned_booking(int(booking.id))
        except Exception:
            logger.warning(
                "Dispatch seed unavailable before Saferpay initialize",
                exc_info=True,
            )

    _lock_booking_row(int(booking.id))

    amount = _client_booking_payable_amount_chf(booking)
    if amount <= Decimal("0"):
        raise ValueError("Montant de réservation invalide pour le paiement")

    value_str = chf_amount_to_saferpay_value_str(amount)
    if int(value_str) < _SAFERPAY_MIN_AMOUNT_CENTS:
        raise ValueError(
            f"Montant minimum ({_SAFERPAY_MIN_AMOUNT_CENTS} centimes) non atteint"
        )

    customer_id = os.environ["SAFERPAY_CUSTOMER_ID"].strip()
    terminal_id = os.environ["SAFERPAY_TERMINAL_ID"].strip()

    payment_query = Payment.query.filter_by(
        booking_id=booking.id,
        payment_provider="saferpay",
        client_id=client.id,
    ).filter(Payment.status == PaymentStatus.PENDING)
    if _supports_row_locking():
        payment_query = payment_query.with_for_update()
    payment_row = payment_query.order_by(Payment.id.desc()).first()

    if payment_row is None:
        payment_row = Payment()
        payment_row.amount = float(amount)
        payment_row.method = "credit_card"
        payment_row.user_id = user.id
        payment_row.client_id = client.id
        payment_row.booking_id = booking.id
        payment_row.status = PaymentStatus.PENDING
        payment_row.payment_provider = "saferpay"
        db.session.add(payment_row)
        db.session.flush()
    else:
        payment_row.amount = float(amount)
        payment_row.saferpay_token = None
        payment_row.saferpay_transaction_id = None

    payment_row.saferpay_notify_key = secrets.token_urlsafe(32)
    db.session.flush()

    if return_url_override and str(return_url_override).strip():
        raw = str(return_url_override).strip()
        parsed0 = urlparse(raw)
        if parsed0.scheme in ("http", "https") and parsed0.netloc:
            _ = validate_return_url_override(raw)
            success_url, fail_url, abort_url = _return_urls_with_outcome_override(
                raw,
                booking_id=booking.id,
                payment_id=payment_row.id,
            )
        else:
            if parsed0.scheme and parsed0.scheme not in ("http", "https"):
                logger.info(
                    "Client Saferpay: return_url appli (schéma %s) — ReturnUrls = HTTPS",
                    parsed0.scheme,
                )
            success_url, fail_url, abort_url = default_saferpay_return_urls(
                booking_id=booking.id,
                payment_id=payment_row.id,
            )
    else:
        success_url, fail_url, abort_url = default_saferpay_return_urls(
            booking_id=booking.id,
            payment_id=payment_row.id,
        )

    order_id = f"L{booking.id}P{payment_row.id}"[:40]

    payload: dict[str, Any] = {
        "RequestHeader": _request_header(customer_id),
        "TerminalId": terminal_id,
        "Payment": {
            "Amount": {"Value": value_str, "CurrencyCode": "CHF"},
            "OrderId": order_id,
            "Description": "Transport LIRIE",
        },
        "Payer": {"LanguageCode": (os.getenv("SAFERPAY_PAYER_LANGUAGE") or "fr").strip()},
        "ReturnUrls": {
            "Success": success_url,
            "Fail": fail_url,
            "Abort": abort_url,
        },
    }

    api_prefix = saferpay_public_api_prefix()
    if api_prefix:
        nk = payment_row.saferpay_notify_key or ""
        payload["Notification"] = {
            "SuccessNotifyUrl": (
                f"{api_prefix}/payments/saferpay/notify/success"
                f"?paymentId={payment_row.id}&k={nk}"
            ),
            "FailNotifyUrl": (
                f"{api_prefix}/payments/saferpay/notify/fail"
                f"?paymentId={payment_row.id}&k={nk}"
            ),
        }

    st, data, raw = saferpay_post_json("Payment/v1/PaymentPage/Initialize", payload)
    if st != HTTPStatus.OK or not data:
        db.session.rollback()
        raise RuntimeError(
            f"Saferpay Initialize refusé (HTTP {st}): {(raw or '')[:400]}"
        )

    token = (data.get("Token") or "").strip()
    redirect_url = (data.get("RedirectUrl") or "").strip()
    if not token or not redirect_url:
        db.session.rollback()
        raise RuntimeError("Réponse Saferpay incomplète (Token / RedirectUrl)")

    payment_row.saferpay_token = token
    db.session.commit()

    logger.info(
        "Saferpay PaymentPage Initialize",
        extra={
            "booking_id": booking.id,
            "payment_id": payment_row.id,
            "api_base": saferpay_api_base_url(),
        },
    )

    return {
        "payment_id": payment_row.id,
        "booking_id": booking.id,
        "redirect_url": redirect_url,
        "order_id": order_id,
        "payment_provider": "saferpay",
        "payment_status": "pending_verification",
        "payment_amount": float(amount),
        "currency": "CHF",
    }


def try_finalize_saferpay_by_payment_id(
    *,
    payment_id: int,
    notify_key: str | None,
) -> dict[str, Any]:
    """Utilisé par les webhooks HTTP GET Saferpay (notification)."""
    row = db.session.get(Payment, payment_id)
    if row is None:
        return {"ok": False, "reason": "not_found"}
    if getattr(row, "payment_provider", None) != "saferpay":
        return {"ok": False, "reason": "not_found"}
    stored_key = getattr(row, "saferpay_notify_key", None)
    if stored_key and (not notify_key or notify_key != stored_key):
        return {"ok": False, "reason": "bad_key"}
    try:
        out = finalize_saferpay_payment(row)
    except Exception:
        logger.exception("try_finalize_saferpay_by_payment_id payment_id=%s", payment_id)
        return {"ok": False, "reason": "error"}
    return {"ok": True, **out}
