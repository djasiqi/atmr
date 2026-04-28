"""Retour d'urgence cote client (fin de rendez-vous), aligne sur dispatch-now."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

from ext import db
from models.booking import Booking
from models.company import Company
from models.enums import BookingStatus

logger = logging.getLogger(__name__)

_OUTBOUND_DONE = frozenset(
    {
        BookingStatus.COMPLETED,
        BookingStatus.RETURN_COMPLETED,
    }
)


def _booking_status(booking: Booking) -> BookingStatus | None:
    st = getattr(booking, "status", None)
    if st is None:
        return None
    if isinstance(st, BookingStatus):
        return st
    sv = str(getattr(st, "value", st)).strip().upper()
    for m in BookingStatus:
        if m.value.upper() == sv:
            return m
    return None


def _return_terminal(st: BookingStatus | None) -> bool:
    if st is None:
        return False
    return st in (
        BookingStatus.RETURN_COMPLETED,
        BookingStatus.COMPLETED,
        BookingStatus.CANCELED,
    )


def apply_client_urgent_return_dispatch(
    *,
    outbound: Booking,
    minutes_offset: int = 15,
) -> tuple[bool, str | None, dict[str, Any] | None]:
    """Fixe le retour a maintenant + offset, confirme l'heure, tente l'assignation.

    ``outbound`` : course aller terminee (transport), avec segment retour lie.
    """
    if bool(getattr(outbound, "is_return", False)):
        return False, "booking_must_be_outbound", None

    ret = getattr(outbound, "return_trip", None)
    if ret is None:
        ret = (
            Booking.query.filter_by(
                parent_booking_id=outbound.id,
                is_return=True,
            ).first()
        )
    if ret is None:
        return False, "return_segment_missing", None

    out_st = _booking_status(outbound)
    if out_st not in _OUTBOUND_DONE:
        return False, "outbound_not_completed", None

    ret_st = _booking_status(ret)
    if ret_st is None:
        return False, "return_status_unknown", None
    if _return_terminal(ret_st):
        return False, "return_already_finished", None
    if ret_st in (BookingStatus.EN_ROUTE, BookingStatus.IN_PROGRESS):
        return False, "return_already_started", None

    cid_obj = getattr(ret, "company_id", None)
    try:
        cid = int(cid_obj) if cid_obj is not None else 0
    except (TypeError, ValueError):
        cid = 0
    if cid <= 0:
        return False, "company_missing_on_return", None

    from shared.time_utils import now_local

    now = now_local()
    ret.scheduled_time = now + timedelta(minutes=max(1, int(minutes_offset)))
    ret.time_confirmed = True

    if ret_st in (BookingStatus.PENDING, BookingStatus.CANCELED):
        ret.status = BookingStatus.ACCEPTED

    db.session.add(ret)
    db.session.commit()
    db.session.refresh(ret)

    company = db.session.get(Company, cid)
    assigned_driver = None
    if bool(getattr(company, "dispatch_enabled", True)):
        from application.companies.request_dispatch import (
            RequestDispatchCommand,
            RequestDispatchUseCase,
        )
        from application.companies.reservations.dispatch_now import DispatchNowUseCase
        from application.events.event_bus import publish_event
        from infrastructure.dispatch.dispatch_now_adapters import (
            DispatchNowAssignmentsApplierAdapter,
            DispatchNowProblemBuilderAdapter,
            DispatchNowUrgentAssignerAdapter,
        )
        from infrastructure.dispatch.settings_adapter import Settings
        from repositories.company_repository import CompanyRepository

        today_str = now_local().strftime("%Y-%m-%d")
        uc = DispatchNowUseCase(
            builder=DispatchNowProblemBuilderAdapter(),
            assigner=DispatchNowUrgentAssignerAdapter(),
            applier=DispatchNowAssignmentsApplierAdapter(),
        )
        uc_result = uc.execute(
            company_id=cid,
            booking_id=int(ret.id),
            today_str=today_str,
            settings=Settings(),
        )

        if uc_result.should_fallback_trigger_dispatch:
            RequestDispatchUseCase(
                company_repo=CompanyRepository(),
                publish_event_fn=publish_event,
            ).execute(
                RequestDispatchCommand(
                    company_id=cid,
                    action="update",
                    reason="booking_update",
                )
            )

        if uc_result.assigned_driver_id:
            from repositories.driver_repository import DriverRepository

            ad = DriverRepository().find_model_by_id(uc_result.assigned_driver_id)
            if ad is not None:
                assigned_driver = ad

    db.session.refresh(ret)

    from services.reservations_summary_cache import invalidate_summary_cache_for_booking

    invalidate_summary_cache_for_booking(cid, ret)

    sched_iso = ret.scheduled_time.isoformat() if ret.scheduled_time else None
    payload: dict[str, Any] = {
        "return_booking_id": ret.id,
        "scheduled_time": sched_iso,
        "time_confirmed": bool(ret.time_confirmed),
        "status": (
            ret.status.value
            if hasattr(ret.status, "value")
            else str(ret.status or "")
        ).lower(),
    }
    if assigned_driver is not None:
        payload["assigned_driver_id"] = int(assigned_driver.id)
    logger.info(
        "client_urgent_return_dispatch outbound_id=%s return_id=%s company_id=%s",
        outbound.id,
        ret.id,
        cid,
    )
    return True, None, payload
