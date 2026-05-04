"""Annulation automatique des réservations en attente de paiement client (délai 15 min)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import exists, func, or_

from application.bookings.cancellation_rules import (
    compute_cancellation_fields,
    log_cancellation_persisted,
)
from ext import db
from models import Booking, Payment
from models.enums import BookingStatus, PaymentStatus

logger = logging.getLogger(__name__)

CLIENT_ONLINE_PAYMENT_GRACE_MINUTES = 15


def expire_awaiting_client_payment_bookings(
    *,
    now: datetime | None = None,
    grace_minutes: int = CLIENT_ONLINE_PAYMENT_GRACE_MINUTES,
) -> int:
    """Annule (CANCELED) les réservations AWAITING_CLIENT_PAYMENT non payées après le délai.

    Les réservations avec au moins un paiement ``COMPLETED`` sont conservées (webhook / course).

    Appel typiquement avant lecture des listes réservations côté client (pas de cron requis).

    Returns:
        Nombre de réservations annulées (commits effectués par réservation).
    """
    now_utc = now or datetime.now(UTC)
    if getattr(now_utc, "tzinfo", None) is None:
        now_utc = now_utc.replace(tzinfo=UTC)

    threshold = now_utc - timedelta(minutes=grace_minutes)

    has_completed_payment = exists().where(
        Payment.booking_id == Booking.id,
        Payment.status == PaymentStatus.COMPLETED,
    )

    # Lot 1 Saferpay : ne pas annuler si session ou trace Assert encore présente
    # (complété en prod par réconciliation / runbook — voir plan §11.1).
    has_active_saferpay_pending = exists().where(
        Payment.booking_id == Booking.id,
        Payment.payment_provider == "saferpay",
        Payment.status == PaymentStatus.PENDING,
        or_(
            Payment.saferpay_token.isnot(None),
            Payment.saferpay_transaction_id.isnot(None),
        ),
    )

    rows = (
        Booking.query.filter(Booking.status == BookingStatus.AWAITING_CLIENT_PAYMENT)
        .filter(
            func.lower(func.coalesce(Booking.billed_to_type, "patient")) == "patient"
        )
        .filter(Booking.created_at.isnot(None))
        .filter(Booking.created_at <= threshold)
        .filter(~has_completed_payment)
        .filter(~has_active_saferpay_pending)
        .all()
    )

    cancelled = 0
    for booking in rows:
        try:
            cancel_fields = compute_cancellation_fields(
                reason_code="PAYMENT_TIMEOUT",
                reason_text="Délai de 15 minutes pour le paiement en ligne dépassé.",
                cancelled_by_role="system",
                now=now_utc,
                booking=booking,
                status_at_cancel="AWAITING_CLIENT_PAYMENT",
            )
            for key, val in cancel_fields.items():
                setattr(booking, key, val)
            booking.update_status(BookingStatus.CANCELED)
            log_cancellation_persisted(booking, cancel_fields)
            db.session.commit()
            cancelled += 1
        except Exception:
            db.session.rollback()
            logger.exception(
                "expire_awaiting_client_payment_bookings: échec booking_id=%s",
                getattr(booking, "id", None),
            )
    return cancelled
