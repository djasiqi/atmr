"""Éligibilité facturation période — sémantique autoritaire ``period-preview`` (R07-OPP-01).

Source de vérité unique pour :
- ``build_period_invoice_preview``
- ``load_eligible_bookings_for_opportunity``
- « Contrôle facturation » institution
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import or_
from sqlalchemy.orm import Query

from models import Booking
from models.enums import BookingStatus

CALENDAR_MONTHS = 12

_BILLABLE_TERMINAL_STATUSES = (
    BookingStatus.COMPLETED.value,
    BookingStatus.RETURN_COMPLETED.value,
)


def period_bounds(period_year: int, period_month: int) -> tuple[datetime, datetime]:
    """Bornes ``[start, end)`` calendaires d'une période de facturation."""
    start_date = datetime(period_year, period_month, 1)
    if period_month == CALENDAR_MONTHS:
        end_date = datetime(period_year + 1, 1, 1)
    else:
        end_date = datetime(period_year, period_month + 1, 1)
    return start_date, end_date


def sqlalchemy_status_filter_for_billing_period(
    billed_to_type: str | None,
) -> Any:
    """Clause statuts alignée ``BookingRepository.find_models_eligible_for_billing_period_by_company_and_client``."""
    status_filter = Booking.status.in_(_BILLABLE_TERMINAL_STATUSES)
    if billed_to_type in ("patient", "clinic"):
        canceled_eligible = (
            (Booking.status == BookingStatus.CANCELED.value)
            & (Booking.amount > 0)
            & (
                (Booking.is_cancellation_billable == True)  # noqa: E712
                | (
                    Booking.billing_override_reason.isnot(None)
                    & (Booking.billing_override_reason != "")
                )
            )
        )
        status_filter = or_(status_filter, canceled_eligible)
    return status_filter


def apply_patient_return_parent_sql_filter(query: Query) -> Query:
    """Exclut retour dont l'aller est annulé (facturation patient directe)."""
    from sqlalchemy.orm import aliased

    parent_alias = aliased(Booking)
    return query.outerjoin(
        parent_alias, parent_alias.id == Booking.parent_booking_id
    ).filter(
        or_(
            Booking.is_return == False,  # noqa: E712
            parent_alias.id.is_(None),
            parent_alias.status != BookingStatus.CANCELED.value,
        )
    )


def _booking_status_eligible(booking: Any, *, billed_to_type: str) -> bool:
    status = getattr(booking, "status", None)
    if status in _BILLABLE_TERMINAL_STATUSES:
        return True
    if billed_to_type not in ("patient", "clinic"):
        return False
    if status != BookingStatus.CANCELED.value:
        return False
    try:
        amount = float(getattr(booking, "amount", 0) or 0)
    except (TypeError, ValueError):
        amount = 0.0
    if amount <= 0:
        return False
    if bool(getattr(booking, "is_cancellation_billable", False)):
        return True
    reason = getattr(booking, "billing_override_reason", None)
    return bool(reason and str(reason).strip())


def _booking_in_period(
    booking: Any,
    *,
    start_date: datetime,
    end_date: datetime,
) -> bool:
    scheduled = getattr(booking, "scheduled_time", None)
    if scheduled is None:
        return False
    if not isinstance(scheduled, datetime):
        return False
    return start_date <= scheduled < end_date


def _booking_passes_patient_return_parent_rule(
    booking: Any,
    *,
    parent_by_id: dict[int, Any],
) -> bool:
    if bool(getattr(booking, "is_return", False)) is False:
        return True
    parent_id = getattr(booking, "parent_booking_id", None)
    if parent_id is None:
        return True
    try:
        pid = int(parent_id)
    except (TypeError, ValueError):
        return True
    parent = parent_by_id.get(pid)
    if parent is None:
        return True
    return getattr(parent, "status", None) != BookingStatus.CANCELED.value


def booking_matches_period_preview_eligibility(
    booking: Any,
    *,
    company_id: int,
    period_year: int,
    period_month: int,
    billed_to_type: str | None = None,
    parent_by_id: dict[int, Any] | None = None,
) -> bool:
    """Vérifie qu'un booking isolé respecte le filet period-preview (hors claim/FK)."""
    try:
        if int(getattr(booking, "company_id", 0) or 0) != int(company_id):
            return False
    except (TypeError, ValueError):
        return False

    btype = str(getattr(booking, "billed_to_type", None) or "").lower().strip()
    expected = str(billed_to_type or btype).lower().strip()
    if btype != expected:
        return False

    start_date, end_date = period_bounds(period_year, period_month)
    if not _booking_in_period(booking, start_date=start_date, end_date=end_date):
        return False

    if not _booking_status_eligible(booking, billed_to_type=btype):
        return False

    if btype == "patient":
        parents = parent_by_id or {}
        if not _booking_passes_patient_return_parent_rule(
            booking, parent_by_id=parents
        ):
            return False

    return True


def _ensure_parent_bookings_loaded(
    bookings: list[Any],
    *,
    parent_by_id: dict[int, Any],
) -> None:
    missing: set[int] = set()
    for booking in bookings:
        parent_id = getattr(booking, "parent_booking_id", None)
        if parent_id is None:
            continue
        try:
            pid = int(parent_id)
        except (TypeError, ValueError):
            continue
        if pid not in parent_by_id:
            missing.add(pid)
    if not missing:
        return
    for parent in Booking.query.filter(Booking.id.in_(missing)).all():
        parent_by_id[int(parent.id)] = parent


def filter_bookings_period_preview_eligible(
    bookings: list[Any],
    *,
    company_id: int,
    period_year: int,
    period_month: int,
    billed_to_type: str = "patient",
    require_open_for_invoice: bool = True,
) -> list[Any]:
    """Filtre in-memory + claim/FK — sémantique ``build_period_invoice_preview`` patient."""
    if not bookings:
        return []

    parent_by_id: dict[int, Any] = {
        int(b.id): b for b in bookings if getattr(b, "id", None)
    }
    _ensure_parent_bookings_loaded(bookings, parent_by_id=parent_by_id)

    eligible: list[Any] = []
    for booking in bookings:
        btype = str(getattr(booking, "billed_to_type", None) or billed_to_type)
        if booking_matches_period_preview_eligibility(
            booking,
            company_id=company_id,
            period_year=period_year,
            period_month=period_month,
            billed_to_type=btype,
            parent_by_id=parent_by_id,
        ):
            eligible.append(booking)

    if not require_open_for_invoice:
        return eligible

    from application.invoices.round_trip_billing_lock import (
        filter_bookings_open_for_new_invoice_line,
    )

    return filter_bookings_open_for_new_invoice_line(eligible)
