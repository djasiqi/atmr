"""Liste des bookings en contrôle facturation institution (compat + détail)."""

from __future__ import annotations

from typing import Any

from application.institutions.billing_control.presentation import (
    serialize_billing_control_booking,
)
from application.institutions.billing_control.query import (
    BillingControlQueryParams,
    query_billing_control_bookings,
)
from models import Booking


def list_billing_control_bookings(
    institution_id: int,
    *,
    control_status_filter: str | None = None,
    period_year: int | None = None,
    period_month: int | None = None,
    payer_type: str | None = None,
    transport_company_id: int | None = None,
    patient_id: int | None = None,
    page: int = 1,
    page_size: int = 50,
):
    """Compat historique — délègue à ``query_billing_control_bookings``."""
    params = BillingControlQueryParams(
        period_year=period_year,
        period_month=period_month,
        control_status=control_status_filter,
        payer_type=payer_type,
        transport_company_id=transport_company_id,
        patient_id=patient_id,
        page=page,
        page_size=page_size,
    )
    return query_billing_control_bookings(institution_id, params)


def booking_control_detail(
    booking: Booking, *, institution_id: int | None = None
) -> dict[str, Any]:
    if institution_id is not None:
        from application.institutions.billing_control.query import (
            booking_control_detail_payload,
        )

        return booking_control_detail_payload(booking, institution_id=institution_id)
    return serialize_billing_control_booking(booking)
