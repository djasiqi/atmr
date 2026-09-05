"""Filtres d'éligibilité facturation clinique S2 (transports portail institution inclus)."""

from __future__ import annotations

from sqlalchemy import and_, exists, func, or_
from sqlalchemy.orm import aliased
from sqlalchemy.sql.elements import ColumnElement

from models import Booking, Client, Company
from models.enums import BookingCreatedVia, BookingStatus

_COMPLETED_FOR_S2 = frozenset(
    {
        BookingStatus.COMPLETED.value,
        BookingStatus.RETURN_COMPLETED.value,
        "COMPLETED",
        "RETURN_COMPLETED",
    }
)
_CANCELED_FOR_S2 = frozenset(
    {
        BookingStatus.CANCELED.value,
        "CANCELED",
        "CANCELLED",
    }
)


def institution_portal_clinic_booking_matches(
    clinic_company_id: int,
    transport_company_id: int,
) -> ColumnElement[bool]:
    """Bookings institution `billed_to_type=clinic` rattachés à cette clinique (même si `billed_to_company_id` erroné)."""
    ccid = int(clinic_company_id)
    carrier_id = int(transport_company_id)
    InstClient = aliased(Client)
    name_match = exists().where(
        Company.id == ccid,
        InstClient.id == Booking.client_id,
        InstClient.is_institution.is_(True),
        func.lower(Company.name) == func.lower(InstClient.institution_name),
    )
    return and_(
        Booking.created_via == BookingCreatedVia.INSTITUTION_PORTAL,
        Booking.billed_to_type == "clinic",
        Booking.company_id == carrier_id,
        exists().where(
            InstClient.id == Booking.client_id,
            InstClient.is_institution.is_(True),
            or_(
                InstClient.default_billed_to_company_id == ccid,
                name_match,
            ),
        ),
    )


def clinic_s2_billed_to_company_predicate(
    clinic_company_id: int,
    transport_company_id: int,
) -> ColumnElement[bool]:
    """Transport facturé à la clinique : champ direct ou portail institution corrigé."""
    ccid = int(clinic_company_id)
    return or_(
        Booking.billed_to_company_id == ccid,
        institution_portal_clinic_booking_matches(ccid, transport_company_id),
    )


def _booking_status_value(booking: object) -> str:
    """Normalise SAEnum / str vers la valeur métier (CANCELED, COMPLETED, …)."""
    raw = getattr(booking, "status", None)
    if raw is None:
        return ""
    return str(getattr(raw, "value", raw) or "").upper().strip()


def cancellation_is_authorized(booking: object) -> bool:
    """C1 : facturable si flag explicite ou override non vide."""
    if getattr(booking, "is_cancellation_billable", False) is True:
        return True
    reason = getattr(booking, "billing_override_reason", None)
    return bool(reason and str(reason).strip())


def cancellation_authorized_sql() -> ColumnElement[bool]:
    return or_(
        Booking.is_cancellation_billable == True,  # noqa: E712
        and_(
            Booking.billing_override_reason.isnot(None),
            Booking.billing_override_reason != "",
        ),
    )


def clinic_canceled_billable_sql() -> ColumnElement[bool]:
    """Annulation clinique éligible : payeur déjà filtré, pas de ClientStay."""
    return and_(
        Booking.status == BookingStatus.CANCELED.value,
        Booking.amount > 0,
        cancellation_authorized_sql(),
        Booking.is_return == False,  # noqa: E712
    )


def booking_is_clinic_canceled_c1_eligible(booking: object) -> bool:
    """Revalidation in-memory C1 d'un segment (après expansion A/R)."""
    status = _booking_status_value(booking)
    if status not in _CANCELED_FOR_S2:
        return False
    try:
        amount = float(getattr(booking, "amount", 0) or 0)
    except (TypeError, ValueError):
        amount = 0.0
    if amount <= 0:
        return False
    if bool(getattr(booking, "is_return", False)):
        return False
    payer = str(getattr(booking, "billed_to_type", "") or "").lower().strip()
    if payer and payer != "clinic":
        return False
    return cancellation_is_authorized(booking)


def filter_clinic_s2_financial_segments(bookings: list) -> list:
    """Garde les courses terminées et les annulations qui passent C1.

    Un pair A/R peut rester en mémoire ; il ne devient pas un segment financier.
    """
    kept: list = []
    for booking in bookings:
        status = _booking_status_value(booking)
        if status in _COMPLETED_FOR_S2:
            kept.append(booking)
            continue
        if booking_is_clinic_canceled_c1_eligible(booking):
            kept.append(booking)
    return kept
