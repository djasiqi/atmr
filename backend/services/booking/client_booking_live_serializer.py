"""Enrichit les réservations sérialisées pour le portail client (position chauffeur, ETA)."""

from __future__ import annotations

import math
from typing import Any

from application.drivers.get_driver_bookings_eta import GetDriverBookingsETAUseCase
from infrastructure.dispatch.eta_calculator import get_eta_seconds_fn
from models.booking import Booking
from models.enums import BookingStatus
from shared.time_utils import iso_utc_z, now_local, to_utc_from_db

_LIVE_STATUSES = frozenset(
    {
        BookingStatus.ASSIGNED,
        BookingStatus.EN_ROUTE,
        BookingStatus.IN_PROGRESS,
    }
)


def _parse_booking_status(booking: Any) -> BookingStatus | None:
    raw = getattr(booking.status, "value", booking.status)
    if raw is None:
        return None
    sv = str(raw).strip().upper()
    for mem in BookingStatus:
        if mem.value.upper() == sv:
            return mem
    return None


def _minutes_from_seconds(sec: int | None) -> int | None:
    if sec is None:
        return None
    return max(0, math.ceil(float(sec) / 60.0))


def enrich_booking_dict_with_client_live(
    booking: Any, data: dict[str, Any]
) -> dict[str, Any]:
    """Ajoute driver_live_* , eta_minutes, estimated_* pour suivi client (carte / texte)."""
    st = _parse_booking_status(booking)
    if st is None or st not in _LIVE_STATUSES:
        return data

    driver = getattr(booking, "driver", None)
    if not driver:
        return data

    dlat = getattr(driver, "latitude", None)
    dlon = getattr(driver, "longitude", None)

    out = {**data}
    if dlat is not None and dlon is not None:
        try:
            out["driver_live_latitude"] = float(dlat)
            out["driver_live_longitude"] = float(dlon)
        except (TypeError, ValueError):
            out["driver_live_latitude"] = None
            out["driver_live_longitude"] = None
    else:
        out["driver_live_latitude"] = None
        out["driver_live_longitude"] = None

    uc = GetDriverBookingsETAUseCase(
        eta_seconds_fn=get_eta_seconds_fn(),
        now_local_fn=now_local,
    )
    resp = uc.execute(
        driver_lat=float(dlat) if dlat is not None else None,
        driver_lon=float(dlon) if dlon is not None else None,
        bookings=[booking],
    )

    if not resp.bookings:
        out["has_driver_live_gps"] = bool(resp.has_gps)
        return out

    item = resp.bookings[0]
    out["has_driver_live_gps"] = bool(resp.has_gps)

    if st == BookingStatus.IN_PROGRESS:
        out["eta_minutes"] = _minutes_from_seconds(item.eta_to_dropoff_seconds)
        out["estimated_dropoff_arrival"] = item.estimated_arrival_dropoff
        out["estimated_pickup_arrival"] = None
        out["client_live_eta_leg"] = "dropoff"
    else:
        out["eta_minutes"] = _minutes_from_seconds(item.eta_to_pickup_seconds)
        out["estimated_pickup_arrival"] = item.estimated_arrival
        out["estimated_dropoff_arrival"] = None
        out["client_live_eta_leg"] = "pickup"

    return out


def _return_booking_client_summary(ret: Any) -> dict[str, Any]:
    st = getattr(ret, "status", None)
    sv = getattr(st, "value", st)
    sched = getattr(ret, "scheduled_time", None)
    return {
        "id": int(getattr(ret, "id", 0) or 0),
        "status": str(sv).lower() if sv is not None else None,
        "scheduled_time": (
            iso_utc_z(to_utc_from_db(sched)) if sched is not None else None
        ),
        "time_confirmed": bool(getattr(ret, "time_confirmed", True)),
    }


def _portal_ceiling_by_booking_id(
    booking_ids: list[int],
) -> dict[int, tuple[float | None, float | None]]:
    """Map booking_id → (estimated_amount_snapshot, maximum_accepted_amount_snapshot).

    Source : événement ``BOOKING_CREATED`` (plafond figé à la création PORTAL DV).
    """
    if not booking_ids:
        return {}
    from models.client_booking_contract_event import (
        EVENT_BOOKING_CREATED,
        ClientBookingContractEvent,
    )

    rows = (
        ClientBookingContractEvent.query.filter(
            ClientBookingContractEvent.booking_id.in_(booking_ids),
            ClientBookingContractEvent.event_type == EVENT_BOOKING_CREATED,
        )
        .order_by(ClientBookingContractEvent.id.asc())
        .all()
    )
    out: dict[int, tuple[float | None, float | None]] = {}
    for ev in rows:
        bid = int(getattr(ev, "booking_id", 0) or 0)
        if bid <= 0 or bid in out:
            continue
        est = getattr(ev, "estimated_amount_snapshot", None)
        mx = getattr(ev, "maximum_accepted_amount_snapshot", None)
        out[bid] = (
            float(est) if est is not None else None,
            float(mx) if mx is not None else None,
        )
    return out


def _pending_portal_offer_booking_ids(booking_ids: list[int]) -> set[int]:
    """Bookings ayant une ``PortalCarrierOffer`` active (offered) non expirée côté liste client."""
    if not booking_ids:
        return set()
    try:
        from models.portal_carrier_offer import (
            OFFER_STATUS_OFFERED,
            PortalCarrierOffer,
        )
    except Exception:
        return set()
    rows = (
        PortalCarrierOffer.query.filter(
            PortalCarrierOffer.booking_id.in_(booking_ids),
            PortalCarrierOffer.status == OFFER_STATUS_OFFERED,
        )
        .with_entities(PortalCarrierOffer.booking_id)
        .all()
    )
    return {int(r[0]) for r in rows if r[0] is not None}


def _portal_contractual_by_booking_id(
    booking_ids: list[int],
) -> dict[int, float]:
    """Montant contractuel : formation 7B.5 ou confirmation DV."""
    if not booking_ids:
        return {}
    out: dict[int, float] = {}
    try:
        from models.portal_transport_contract_formed import (
            PortalTransportContractFormed,
        )

        for row in PortalTransportContractFormed.query.filter(
            PortalTransportContractFormed.booking_id.in_(booking_ids)
        ).all():
            bid = int(row.booking_id)
            out[bid] = float(row.carrier_quote)
    except Exception:
        pass
    try:
        from models.portal_client_transport_confirmation import (
            PortalClientTransportConfirmation,
        )

        for row in PortalClientTransportConfirmation.query.filter(
            PortalClientTransportConfirmation.booking_id.in_(booking_ids)
        ).all():
            bid = int(row.booking_id)
            if bid not in out:
                out[bid] = float(row.contractual_amount)
    except Exception:
        pass
    return out


def enrich_client_bookings_list(bookings: list[Any]) -> list[dict[str, Any]]:
    booking_ids = [
        int(getattr(b, "id", 0) or 0)
        for b in bookings
        if getattr(b, "id", None) is not None
    ]
    ids = [i for i in booking_ids if i > 0]
    ceilings = _portal_ceiling_by_booking_id(ids)
    pending_offer_ids = _pending_portal_offer_booking_ids(ids)
    contractual = _portal_contractual_by_booking_id(ids)

    out: list[dict[str, Any]] = []
    for b in bookings:
        ser = b.serialize
        base: dict[str, Any] = ser if isinstance(ser, dict) else {}
        data = enrich_booking_dict_with_client_live(b, base)
        bid = int(getattr(b, "id", 0) or 0)
        if bid in ceilings:
            est, mx = ceilings[bid]
            if est is not None:
                data["estimated_amount_snapshot"] = est
            if mx is not None:
                data["maximum_accepted_amount"] = mx
                data["maximum_accepted_amount_snapshot"] = mx
        if bid in contractual:
            data["contractual_amount"] = contractual[bid]
        if bid in pending_offer_ids:
            data["has_pending_portal_offer"] = True
        if not bool(getattr(b, "is_return", False)):
            ret = getattr(b, "return_trip", None)
            if ret is None and (
                bool(getattr(b, "is_round_trip", False))
                or bool(base.get("has_return"))
                or bool(base.get("is_round_trip"))
            ):
                ret = Booking.query.filter_by(
                    parent_booking_id=getattr(b, "id", None),
                    is_return=True,
                ).first()
            if ret is not None:
                data["return_booking"] = _return_booking_client_summary(ret)
        out.append(data)
    return out
