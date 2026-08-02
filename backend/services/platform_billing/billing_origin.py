"""Pose, backfill et correction auditée de billing_origin."""

from __future__ import annotations

import logging
from typing import Any

from ext import db
from models.enums import (
    BillingOriginSource,
    BookingBillingOrigin,
    BookingCreatedVia,
    PlatformStatementStatus,
)
from models.platform_billing import BookingBillingOriginAudit, PlatformInvoice
from services.admin_booking_billing_kernel import classify_booking_source

logger = logging.getLogger(__name__)

_LOCKED_STATEMENT = {
    PlatformStatementStatus.VALIDATED.value,
    PlatformStatementStatus.LOCKED.value,
}


def resolve_billing_origin_for_create(
    *,
    created_via: str | BookingCreatedVia | None = None,
    explicit_origin: str | None = None,
    is_institution_flow: bool = False,
    is_admin: bool = False,
) -> tuple[str, str, str | None]:
    """Retourne (origin, source, reason) pour une création.

    created_via reste technique ; billing_origin est commercial.
    """
    if explicit_origin:
        return (
            explicit_origin,
            BillingOriginSource.EXPLICIT_AT_CREATION.value,
            "EXPLICIT_USE_CASE",
        )
    via = created_via.value if hasattr(created_via, "value") else created_via
    if is_institution_flow or via == BookingCreatedVia.INSTITUTION_PORTAL.value:
        return (
            BookingBillingOrigin.LIRIE_MARKETPLACE.value,
            BillingOriginSource.EXPLICIT_AT_CREATION.value,
            "INSTITUTION_OR_PORTAL",
        )
    if is_admin or via is None:
        if is_admin:
            return (
                BookingBillingOrigin.ADMIN_CREATED.value,
                BillingOriginSource.EXPLICIT_AT_CREATION.value,
                "ADMIN_DEFAULT",
            )
    if via in (
        BookingCreatedVia.DISPATCHER.value,
        BookingCreatedVia.CLIENT_APP.value,
        BookingCreatedVia.PUBLIC_GUEST.value,
        BookingCreatedVia.API_PARTNER.value,
        BookingCreatedVia.LEGACY.value,
    ):
        return (
            BookingBillingOrigin.OWN_PORTFOLIO.value,
            BillingOriginSource.EXPLICIT_AT_CREATION.value,
            f"CREATED_VIA_{via}",
        )
    return (
        BookingBillingOrigin.UNKNOWN.value,
        BillingOriginSource.EXPLICIT_AT_CREATION.value,
        "UNMAPPED",
    )


def backfill_origin_for_booking(booking: Any) -> tuple[str, str, str]:
    """Backfill déterministe puis heuristique."""
    via = getattr(booking, "created_via", None)
    via_v = via.value if hasattr(via, "value") else via
    if via_v == BookingCreatedVia.INSTITUTION_PORTAL.value:
        return (
            BookingBillingOrigin.LIRIE_MARKETPLACE.value,
            BillingOriginSource.BACKFILL_DETERMINISTIC.value,
            "created_via=institution_portal",
        )
    try:
        tl = booking._get_institution_timeline()
        if tl and tl.get("created_by_name"):
            return (
                BookingBillingOrigin.LIRIE_MARKETPLACE.value,
                BillingOriginSource.BACKFILL_DETERMINISTIC.value,
                "institution_timeline",
            )
    except Exception:
        logger.debug("backfill timeline soft fail", exc_info=True)

    source_code = classify_booking_source(booking)
    mapping = {
        "institution_request": BookingBillingOrigin.LIRIE_MARKETPLACE.value,
        "client_direct": BookingBillingOrigin.OWN_PORTFOLIO.value,
        "company_manual": BookingBillingOrigin.OWN_PORTFOLIO.value,
        "admin_created": BookingBillingOrigin.ADMIN_CREATED.value,
        "unknown_source": BookingBillingOrigin.UNKNOWN.value,
    }
    origin = mapping.get(source_code, BookingBillingOrigin.UNKNOWN.value)
    return (
        origin,
        BillingOriginSource.BACKFILL_HEURISTIC.value,
        f"classify_booking_source={source_code}",
    )


def booking_in_locked_statement(booking_id: int) -> bool:
    from models.platform_billing import PlatformBillingStatementItem

    item = (
        PlatformBillingStatementItem.query.filter_by(booking_id=booking_id)
        .join(PlatformInvoice)
        .filter(PlatformInvoice.statement_status.in_(_LOCKED_STATEMENT))
        .first()
    )
    return item is not None


def correct_billing_origin(
    booking: Any,
    new_origin: str,
    *,
    reason: str,
    author_user_id: int | None,
) -> Any:
    allowed = {o.value for o in BookingBillingOrigin}
    if new_origin not in allowed:
        raise ValueError(f"billing_origin invalide: {new_origin}")
    if not reason or not str(reason).strip():
        raise ValueError("motif requis")
    if booking_in_locked_statement(booking.id):
        raise ValueError(
            "Correction interdite : réservation déjà dans un relevé validé/verrouillé"
        )
    old = getattr(booking, "billing_origin", None)
    booking.billing_origin = new_origin
    booking.billing_origin_source = BillingOriginSource.ADMIN_CORRECTION.value
    booking.billing_origin_reason = reason.strip()
    db.session.add(
        BookingBillingOriginAudit(
            booking_id=booking.id,
            old_value=old,
            new_value=new_origin,
            reason=reason.strip(),
            author_user_id=author_user_id,
        )
    )
    db.session.commit()
    return booking


def apply_origin_on_booking(
    booking: Any,
    *,
    created_via: str | BookingCreatedVia | None = None,
    explicit_origin: str | None = None,
    is_institution_flow: bool = False,
    is_admin: bool = False,
) -> None:
    origin, source, reason = resolve_billing_origin_for_create(
        created_via=created_via or getattr(booking, "created_via", None),
        explicit_origin=explicit_origin,
        is_institution_flow=is_institution_flow,
        is_admin=is_admin,
    )
    booking.billing_origin = origin
    booking.billing_origin_source = source
    booking.billing_origin_reason = reason
