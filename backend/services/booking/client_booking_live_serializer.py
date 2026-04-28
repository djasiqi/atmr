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


def enrich_booking_dict_with_client_live(booking: Any, data: dict[str, Any]) -> dict[str, Any]:
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


def enrich_client_bookings_list(bookings: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for b in bookings:
        ser = b.serialize
        base: dict[str, Any] = ser if isinstance(ser, dict) else {}
        data = enrich_booking_dict_with_client_live(b, base)
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
