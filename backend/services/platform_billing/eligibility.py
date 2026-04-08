"""Éligibilité commission plateforme (règles alignées sur admin_booking_billing_kernel)."""

from __future__ import annotations

from typing import Any

from models import Booking
from models.enums import BookingStatus
from services.admin_booking_billing_kernel import (
    booking_is_executed,
    is_synthetic_demo_booking,
)


def is_cancelled_for_subscription_volume(booking: Booking) -> bool:
    st = booking.status
    key = st.value if hasattr(st, "value") else str(st).upper()
    return key == BookingStatus.CANCELED.value


def is_commissionable_platform(booking: Booking, pilotage_payload: dict[str, Any]) -> bool:
    """Règle V1 stricte : institution + exécuté + eligible + montant > 0 + hors démo + completed_at."""
    if is_synthetic_demo_booking(booking):
        return False
    if not booking_is_executed(booking):
        return False
    if booking.completed_at is None:
        return False
    pl = pilotage_payload
    if pl.get("source_code") != "institution_request":
        return False
    qual = pl.get("qualification") or {}
    if qual.get("state") != "eligible":
        return False
    amt = pl.get("observed_transport_amount")
    if amt is None or float(amt) <= 0:
        return False
    return True
