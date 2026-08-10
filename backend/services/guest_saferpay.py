"""Paiement Saferpay pour réservations invitées (draft Redis → promotion SQL)."""

from __future__ import annotations

import json
import logging
import os
import secrets
import uuid
from contextlib import suppress
from datetime import UTC, datetime
from decimal import Decimal
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from flask import current_app
from itsdangerous import URLSafeTimedSerializer
from sqlalchemy.exc import IntegrityError

from ext import db
from models.booking import Booking
from models.enums import BookingCreatedVia, BookingStatus, PaymentStatus
from models.payment import Payment
from services.saferpay.assert_capture import run_saferpay_paymentpage_assert_capture
from services.saferpay.assert_response_status import (
    SAFERPAY_FINALIZE_ALREADY_COMPLETED,
    SAFERPAY_FINALIZE_ASSERT_FAILED,
    SAFERPAY_FINALIZE_COMPLETED,
    SAFERPAY_FINALIZE_PAYMENT_FAILED,
    SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS,
)
from services.saferpay.config import (
    saferpay_configured,
    saferpay_public_api_prefix,
    saferpay_spec_version,
)
from services.saferpay.http_client import saferpay_post_json
from services.saferpay.money import chf_amount_to_saferpay_value_str
from services.saferpay.return_urls import validate_return_url_override
from shared.guest_booking_constants import GUEST_BOOKING_CUSTOMER_PLACEHOLDER
from shared.time_utils import parse_local_naive

logger = logging.getLogger(__name__)

_SAFERPAY_MIN_AMOUNT_CENTS = 50
_BOOKING_PUBLIC_STATUS_SALT = "booking-status-public-link"


def _request_header(customer_id: str) -> dict[str, Any]:
    return {
        "SpecVersion": saferpay_spec_version(),
        "CustomerId": customer_id,
        "RequestId": uuid.uuid4().hex,
        "RetryIndicator": 0,
    }


def _decimal_amount(value: Any) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value or 0))


def _draft_scheduled_time(payload: dict[str, Any]) -> datetime | None:
    date_s = str(payload.get("date") or "").strip()
    pt = str(payload.get("pickup_time") or "").strip()
    if not date_s:
        return None
    if not pt:
        return parse_local_naive(f"{date_s}T12:00:00")
    if "T" in pt:
        return parse_local_naive(pt)
    if pt.count(":") == 1:
        return parse_local_naive(f"{date_s}T{pt}:00")
    return parse_local_naive(f"{date_s}T{pt}")


def _draft_wheelchair_flags(transport_type: str | None) -> tuple[bool, bool]:
    tt = (transport_type or "assis").strip().lower()
    if tt in {"pmr", "fauteuil", "wheelchair"}:
        return True, False
    return False, False


def _guest_customer_name(payload: dict[str, Any]) -> str:
    fn = str(payload.get("first_name") or "").strip()
    ln = str(payload.get("last_name") or "").strip()
    if fn or ln:
        return f"{fn} {ln}".strip()[:200]
    return GUEST_BOOKING_CUSTOMER_PLACEHOLDER


def _apply_payload_geo_to_guest_booking(
    booking: Booking, payload: dict[str, Any]
) -> None:
    """Recopie le géo du brouillon (calculé au pricing) : requis pour les offres dispatch entreprise."""
    for pkey, attr in (
        ("pickup_lat", "pickup_lat"),
        ("pickup_lon", "pickup_lon"),
        ("dropoff_lat", "dropoff_lat"),
        ("dropoff_lon", "dropoff_lon"),
    ):
        v = payload.get(pkey)
        if v is None:
            continue
        with suppress(TypeError, ValueError):
            setattr(booking, attr, float(v))
    for pkey, attr in (
        ("pickup_geo_unit_id", "pickup_geo_unit_id"),
        ("dropoff_geo_unit_id", "dropoff_geo_unit_id"),
    ):
        v = payload.get(pkey)
        if v is None:
            continue
        with suppress(TypeError, ValueError):
            setattr(booking, attr, int(v))
    for pkey, attr in (
        ("distance_meters", "distance_meters"),
        ("duration_seconds", "duration_seconds"),
    ):
        v = payload.get(pkey)
        if v is None:
            continue
        with suppress(TypeError, ValueError):
            setattr(booking, attr, int(v))


def _issue_public_booking_status_token(booking_id: int) -> str:
    secret_key = current_app.config.get("SECRET_KEY")
    if not secret_key:
        raise RuntimeError("SECRET_KEY non configurée")
    serializer = URLSafeTimedSerializer(secret_key)
    return serializer.dumps(
        {"booking_id": booking_id}, salt=_BOOKING_PUBLIC_STATUS_SALT
    )


def _default_guest_saferpay_return_urls(
    *, guest_booking_id: str
) -> tuple[str, str, str]:
    from services.saferpay.return_urls import (
        _is_loopback_http_url,
        _localhost_return_allowed_for_saferpay,
    )

    base = (os.getenv("SAFERPAY_CHECKOUT_PUBLIC_BASE_URL") or "").strip().rstrip("/")
    if not base:
        base = (os.getenv("CLIENT_WEB_BASE_URL") or "").strip().rstrip("/")
    if not base:
        base = (
            (os.getenv("PUBLIC_BASE_URL") or "http://localhost:3000")
            .strip()
            .rstrip("/")
        )
    q = f"guestBookingId={guest_booking_id}"
    success = f"{base}/guest/payment/saferpay/return?{q}&outcome=success"
    fail = f"{base}/guest/payment/saferpay/return?{q}&outcome=fail"
    abort = f"{base}/guest/payment/saferpay/return?{q}&outcome=abort"
    for u in (success, fail, abort):
        if _is_loopback_http_url(u) and not _localhost_return_allowed_for_saferpay():
            raise ValueError(
                "Saferpay guest: URL de retour localhost souvent refusee. "
                + "Definissez SAFERPAY_CHECKOUT_PUBLIC_BASE_URL "
                + "ou SAFERPAY_ALLOW_LOCALHOST_RETURN=1."
            )
    return success, fail, abort


def _guest_return_urls_with_override(
    *,
    guest_booking_id: str,
    return_url_override: str | None,
) -> tuple[str, str, str]:
    """Build ReturnUrls pour l'appel API Saferpay.

    **Important** : l'API PaymentPage n'accepte en pratique que des URLs
    **http / https** ; un schema d'app (`lirie://`, `exp://`, etc.) est refuse
    (VALIDATION_FAILED sur ReturnUrls). Les liens d'app sont geres par une page
    web HTTPS qui redirige (voir `/guest/payment/saferpay/return` sur le site
    client). On ignore donc les override non-HTTP(S) cote Saferpay et on utilise
    l'URL publique configuree.
    """
    if return_url_override and str(return_url_override).strip():
        raw = str(return_url_override).strip()
        parsed0 = urlparse(raw)
        if parsed0.scheme in ("http", "https") and parsed0.netloc:
            base_override = validate_return_url_override(raw)
            parsed = urlparse(base_override)
            q = dict(parse_qsl(parsed.query, keep_blank_values=True))
            if "guestBookingId" not in q:
                q["guestBookingId"] = str(guest_booking_id)

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

        if parsed0.scheme and parsed0.scheme not in ("http", "https"):
            logger.info(
                "Guest Saferpay: return_url côté appli (schéma %s) — ReturnUrls = HTTPS public",
                parsed0.scheme,
            )
    return _default_guest_saferpay_return_urls(guest_booking_id=guest_booking_id)


def initialize_guest_saferpay(
    *,
    guest_booking_id: str,
    payload: dict[str, Any],
    return_url_override: str | None,
    redis_setex: Any,
    guest_ttl_seconds: int,
) -> dict[str, Any]:
    if not saferpay_configured():
        raise RuntimeError("Saferpay n'est pas configuré sur ce serveur")

    if str(payload.get("status") or "") == "paid" or payload.get("promoted_booking_id"):
        raise ValueError("already_promoted")

    if payload.get("consumed"):
        raise ValueError("guest_booking_consumed")

    amount = _decimal_amount(payload.get("amount") or payload.get("preview_amount"))
    if amount <= Decimal("0"):
        raise ValueError("Montant invité invalide pour le paiement")

    value_str = chf_amount_to_saferpay_value_str(amount)
    if int(value_str) < _SAFERPAY_MIN_AMOUNT_CENTS:
        raise ValueError(
            f"Montant minimum ({_SAFERPAY_MIN_AMOUNT_CENTS} centimes) non atteint"
        )

    customer_id = os.environ["SAFERPAY_CUSTOMER_ID"].strip()
    terminal_id = os.environ["SAFERPAY_TERMINAL_ID"].strip()

    success_url, fail_url, abort_url = _guest_return_urls_with_override(
        guest_booking_id=guest_booking_id,
        return_url_override=return_url_override,
    )

    notify_key = secrets.token_urlsafe(32)
    order_id = f"G{guest_booking_id}"[:40]

    saferpay_payload: dict[str, Any] = {
        "RequestHeader": _request_header(customer_id),
        "TerminalId": terminal_id,
        "Payment": {
            "Amount": {"Value": value_str, "CurrencyCode": "CHF"},
            "OrderId": order_id,
            "Description": "Transport LIRIE (invité)",
        },
        "Payer": {
            "LanguageCode": (os.getenv("SAFERPAY_PAYER_LANGUAGE") or "fr").strip()
        },
        "ReturnUrls": {
            "Success": success_url,
            "Fail": fail_url,
            "Abort": abort_url,
        },
    }

    api_prefix = saferpay_public_api_prefix()
    if api_prefix:
        nk = notify_key
        saferpay_payload["Notification"] = {
            "SuccessNotifyUrl": (
                f"{api_prefix}/payments/saferpay/notify/guest-success"
                f"?guestBookingId={guest_booking_id}&k={nk}"
            ),
            "FailNotifyUrl": (
                f"{api_prefix}/payments/saferpay/notify/guest-fail"
                f"?guestBookingId={guest_booking_id}&k={nk}"
            ),
        }

    st, data, raw = saferpay_post_json(
        "Payment/v1/PaymentPage/Initialize", saferpay_payload
    )
    if st != HTTPStatus.OK or not data:
        raise RuntimeError(
            f"Saferpay Initialize refusé (HTTP {st}): {(raw or '')[:400]}"
        )

    token = (data.get("Token") or "").strip()
    redirect_url = (data.get("RedirectUrl") or "").strip()
    if not token or not redirect_url:
        raise RuntimeError("Réponse Saferpay incomplète (Token / RedirectUrl)")

    payload["saferpay_token"] = token
    payload["saferpay_notify_key"] = notify_key
    payload["saferpay_order_id"] = order_id
    payload["payment_status"] = "pending_verification"
    payload["updated_at"] = datetime.now(UTC).isoformat()

    cache_key = f"public:guest_booking:{guest_booking_id}"
    redis_setex(cache_key, guest_ttl_seconds, json.dumps(payload))

    logger.info(
        "Guest Saferpay PaymentPage Initialize",
        extra={
            "guest_booking_id": guest_booking_id,
            "order_id": order_id,
            "payment_amount_chf": float(amount),
            "has_notify_urls": bool(api_prefix),
        },
    )

    return {
        "guest_booking_id": guest_booking_id,
        "redirect_url": redirect_url,
        "order_id": order_id,
        "payment_provider": "saferpay",
        "payment_status": "pending_verification",
        "payment_amount": float(amount),
        "currency": "CHF",
    }


def _mark_redis_promoted(
    *,
    guest_booking_id: str,
    payload: dict[str, Any],
    booking_id: int,
    public_status_token: str,
    redis_setex: Any,
    promote_ttl_seconds: int = 600,
) -> None:
    payload["consumed"] = True
    payload["promoted_booking_id"] = booking_id
    payload["public_status_token"] = public_status_token
    payload["status"] = "paid"
    payload["updated_at"] = datetime.now(UTC).isoformat()
    cache_key = f"public:guest_booking:{guest_booking_id}"
    redis_setex(cache_key, promote_ttl_seconds, json.dumps(payload))


def promote_guest_booking_after_saferpay(
    *,
    guest_booking_id: str,
    payload: dict[str, Any],
    redis_setex: Any,
    notify_key: str | None = None,
) -> dict[str, Any]:
    """Assert/Capture via token Redis puis création Booking + Payment (idempotent)."""
    if payload.get("promoted_booking_id") and payload.get("public_status_token"):
        bid = int(payload["promoted_booking_id"])
        logger.info(
            "Guest Saferpay promote idempotent (cache déjà promu)",
            extra={
                "guest_booking_id": guest_booking_id,
                "booking_id": bid,
                "outcome": "already_completed_redis",
            },
        )
        return {
            "status": SAFERPAY_FINALIZE_ALREADY_COMPLETED,
            "booking_id": bid,
            "public_status_token": str(payload["public_status_token"]),
            "guest_booking_id": guest_booking_id,
        }

    stored_key = (payload.get("saferpay_notify_key") or "").strip()
    if notify_key and stored_key and notify_key != stored_key:
        logger.warning(
            "Guest Saferpay notify_key refusée",
            extra={
                "guest_booking_id": guest_booking_id,
                "outcome": "forbidden_notify_key",
            },
        )
        return {"status": "forbidden", "detail": "bad_notify_key"}

    session_token = (payload.get("saferpay_token") or "").strip()
    if not session_token:
        raise ValueError("Session Saferpay absente (réinitialiser le paiement)")

    ac = run_saferpay_paymentpage_assert_capture(session_token)
    ac_status = str(ac.get("status") or "")

    if ac_status != SAFERPAY_FINALIZE_COMPLETED:
        if ac_status == SAFERPAY_FINALIZE_PAYMENT_FAILED:
            payload["status"] = "payment_failed"
            payload["updated_at"] = datetime.now(UTC).isoformat()
            redis_setex(
                f"public:guest_booking:{guest_booking_id}",
                max(
                    600,
                    int(
                        os.getenv("PUBLIC_GUEST_BOOKING_TTL_SECONDS", "604800")
                        or 604800
                    ),
                ),
                json.dumps(payload),
            )
            logger.info(
                "Guest Saferpay assert échec paiement (état Redis mis à jour)",
                extra={
                    "guest_booking_id": guest_booking_id,
                    "outcome": "payment_failed",
                },
            )
        else:
            logger.info(
                "Guest Saferpay assert non complété",
                extra={
                    "guest_booking_id": guest_booking_id,
                    "assert_status": ac_status,
                    "outcome": "assert_not_completed",
                },
            )
        return {**ac, "guest_booking_id": guest_booking_id}

    tx_id = (ac.get("tx_id") or "").strip()
    if not tx_id:
        return {
            "status": SAFERPAY_FINALIZE_UNEXPECTED_TX_STATUS,
            "detail": "tx_id manquant après capture",
            "guest_booking_id": guest_booking_id,
        }

    existing_pay = Payment.query.filter_by(saferpay_transaction_id=tx_id).first()
    if existing_pay is not None:
        bk = existing_pay.booking
        if bk is None:
            return {
                "status": SAFERPAY_FINALIZE_ASSERT_FAILED,
                "detail": "booking_manquant",
            }
        token = _issue_public_booking_status_token(int(bk.id))
        _mark_redis_promoted(
            guest_booking_id=guest_booking_id,
            payload=payload,
            booking_id=int(bk.id),
            public_status_token=token,
            redis_setex=redis_setex,
        )
        return {
            "status": SAFERPAY_FINALIZE_ALREADY_COMPLETED,
            "booking_id": bk.id,
            "public_status_token": token,
            "guest_booking_id": guest_booking_id,
        }

    w_need, w_has = _draft_wheelchair_flags(
        str(payload.get("transport_type") or "assis")
    )
    scheduled = _draft_scheduled_time(payload)
    amount_f = float(_decimal_amount(payload.get("amount") or 0))

    booking = Booking()
    booking.customer_name = _guest_customer_name(payload)
    booking.pickup_location = str(payload.get("departure") or "").strip()[:500]
    booking.dropoff_location = str(payload.get("destination") or "").strip()[:500]
    booking.scheduled_time = scheduled
    booking.amount = amount_f
    booking.status = BookingStatus.PENDING
    booking.created_via = BookingCreatedVia.PUBLIC_GUEST
    from services.platform_billing.billing_origin import apply_origin_on_booking

    apply_origin_on_booking(booking, created_via=BookingCreatedVia.PUBLIC_GUEST)
    booking.user_id = None
    booking.client_id = None
    booking.company_id = None
    booking.wheelchair_need = w_need
    booking.wheelchair_client_has = w_has
    _apply_payload_geo_to_guest_booking(booking, payload)

    payment_row = Payment()
    payment_row.amount = amount_f
    payment_row.method = "credit_card"
    payment_row.user_id = None
    payment_row.client_id = None
    payment_row.status = PaymentStatus.COMPLETED
    payment_row.payment_provider = "saferpay"
    payment_row.saferpay_token = session_token
    payment_row.saferpay_transaction_id = tx_id
    payment_row.saferpay_notify_key = stored_key or None

    db.session.add(booking)
    db.session.flush()
    payment_row.booking_id = int(booking.id)
    db.session.add(payment_row)

    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        logger.warning(
            "Guest Saferpay IntegrityError à l'insert (course concurrente ?)",
            extra={
                "guest_booking_id": guest_booking_id,
                "outcome": "integrity_recover",
            },
        )
        existing_pay = Payment.query.filter_by(saferpay_transaction_id=tx_id).first()
        if existing_pay and existing_pay.booking_id:
            bk = existing_pay.booking
            token = _issue_public_booking_status_token(int(bk.id))
            _mark_redis_promoted(
                guest_booking_id=guest_booking_id,
                payload=payload,
                booking_id=int(bk.id),
                public_status_token=token,
                redis_setex=redis_setex,
            )
            return {
                "status": SAFERPAY_FINALIZE_ALREADY_COMPLETED,
                "booking_id": bk.id,
                "public_status_token": token,
                "guest_booking_id": guest_booking_id,
            }
        raise

    token = _issue_public_booking_status_token(int(booking.id))
    _mark_redis_promoted(
        guest_booking_id=guest_booking_id,
        payload=payload,
        booking_id=int(booking.id),
        public_status_token=token,
        redis_setex=redis_setex,
    )

    from services.dispatch.open_booking_offers import (
        refresh_dispatch_offers_after_online_payment,
    )

    refresh_dispatch_offers_after_online_payment(int(booking.id))

    logger.info(
        "Guest Saferpay promu (nouveau booking)",
        extra={
            "guest_booking_id": guest_booking_id,
            "booking_id": booking.id,
            "payment_id": payment_row.id,
            "saferpay_transaction_id": tx_id,
            "outcome": "completed",
        },
    )

    return {
        "status": SAFERPAY_FINALIZE_COMPLETED,
        "booking_id": booking.id,
        "public_status_token": token,
        "guest_booking_id": guest_booking_id,
        "payment_id": payment_row.id,
    }


def try_finalize_guest_saferpay_notify(
    *,
    guest_booking_id: str,
    notify_key: str | None,
    redis_get: Any,
    redis_setex: Any,
) -> dict[str, Any]:
    raw = redis_get(f"public:guest_booking:{guest_booking_id}")
    if not raw:
        logger.info(
            "Guest Saferpay notify sans brouillon Redis",
            extra={"guest_booking_id": guest_booking_id, "outcome": "notify_not_found"},
        )
        return {"ok": False, "reason": "not_found"}
    try:
        payload = json.loads(raw)
    except Exception:
        logger.warning(
            "Guest Saferpay notify cache JSON invalide",
            extra={
                "guest_booking_id": guest_booking_id,
                "outcome": "notify_invalid_cache",
            },
        )
        return {"ok": False, "reason": "invalid_cache"}
    try:
        out = promote_guest_booking_after_saferpay(
            guest_booking_id=guest_booking_id,
            payload=payload,
            redis_setex=redis_setex,
            notify_key=notify_key,
        )
    except ValueError as e:
        logger.info(
            "Guest Saferpay notify finalize refus (validation)",
            extra={
                "guest_booking_id": guest_booking_id,
                "reason": str(e),
                "outcome": "notify_validation_error",
            },
        )
        return {"ok": False, "reason": str(e)}
    except Exception:
        logger.exception(
            "Guest Saferpay notify finalize erreur",
            extra={"guest_booking_id": guest_booking_id, "outcome": "notify_error"},
        )
        return {"ok": False, "reason": "error"}
    st = str(out.get("status") or "")
    logger.info(
        "Guest Saferpay notify finalize",
        extra={
            "guest_booking_id": guest_booking_id,
            "finalize_status": st,
            "booking_id": out.get("booking_id"),
            "outcome": "notify_processed",
        },
    )
    return {"ok": True, **out}
