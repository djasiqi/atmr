"""P1: émission throttlée `eta_changed` après localisation canonique (pas un recalcul par point GPS nu)."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from application.drivers.get_driver_bookings_eta import (
    DriverBookingsEtaResponse,
    GetDriverBookingsETAUseCase,
)
from infrastructure.dispatch.eta_calculator import get_eta_seconds_fn
from models.enums import BookingStatus
from repositories.booking_repository import BookingRepository
from services.monitoring.driver_eta_socket_metrics import (
    inc_eta_changed_emitted,
    inc_eta_changed_skipped,
)
from services.realtime.socketio import emit_driver_event
from shared.time_utils import day_local_bounds, now_local

logger = logging.getLogger(__name__)

_ETA_COMPUTE_MIN_INTERVAL_SEC = 25.0
_ETA_DELTA_THRESHOLD_SEC = 10

_lock = threading.Lock()
_last_compute_monotonic: dict[int, float] = {}
_last_eta_emitted_by_driver: dict[int, dict[int, tuple[int | None, int | None]]] = {}


def _signature_from_response(
    resp: DriverBookingsEtaResponse,
) -> dict[int, tuple[int | None, int | None]]:
    return {
        int(item.id): (item.eta_to_pickup_seconds, item.eta_to_dropoff_seconds)
        for item in resp.bookings
    }


def _significant_delta(
    prev: dict[int, tuple[int | None, int | None]] | None,
    new: dict[int, tuple[int | None, int | None]],
) -> bool:
    if prev is None:
        return True
    if set(prev.keys()) != set(new.keys()):
        return True
    th = _ETA_DELTA_THRESHOLD_SEC
    for bid, (pu, do) in new.items():
        old = prev.get(bid)
        if old is None:
            return True
        opu, odo = old
        if pu is not None and opu is not None and abs(pu - opu) >= th:
            return True
        if do is not None and odo is not None and abs(do - odo) >= th:
            return True
        if (pu is None) != (opu is None) or (do is None) != (odo is None):
            return True
    return False


def _serialize_eta_response(
    resp: DriverBookingsEtaResponse, bookings: list[Any]
) -> dict[str, Any]:
    """Même forme que GET /driver/me/bookings/eta."""
    if not resp.has_gps:
        return {
            "has_gps": False,
            "bookings": [
                {
                    "id": b.id,
                    "duration_seconds": b.duration_seconds,
                    "distance_meters": b.distance_meters,
                    "eta_to_dropoff_seconds": None,
                    "estimated_arrival_dropoff": None,
                }
                for b in bookings
            ],
        }
    return {
        "has_gps": True,
        "driver_position": resp.driver_position,
        "bookings": [
            {
                "id": item.id,
                "eta_to_pickup_seconds": item.eta_to_pickup_seconds,
                "eta_to_dropoff_seconds": item.eta_to_dropoff_seconds,
                "duration_seconds": item.duration_seconds,
                "distance_meters": item.distance_meters,
                "estimated_arrival": item.estimated_arrival,
                "estimated_arrival_dropoff": item.estimated_arrival_dropoff,
            }
            for item in resp.bookings
        ],
    }


def maybe_emit_eta_changed_after_driver_location(
    *,
    driver_id: int,
    driver_lat: float,
    driver_lon: float,
    accept_status: str,
) -> None:
    """Après persistance localisation canonique. No-op si observabilité seule ou pas de missions."""
    if accept_status != "accepted_canonical":
        return

    try:
        today_start, today_end = day_local_bounds(
            now_local().date().strftime("%Y-%m-%d")
        )
        booking_repo = BookingRepository()
        bookings = booking_repo.find_models_by_driver_with_statuses_and_time_range(
            driver_id=driver_id,
            statuses=[
                BookingStatus.ASSIGNED,
                BookingStatus.EN_ROUTE,
                BookingStatus.IN_PROGRESS,
            ],
            start_time=today_start,
            end_time=today_end,
        )
        if not bookings:
            inc_eta_changed_skipped(reason="no_active_bookings")
            return

        now_m = time.monotonic()
        with _lock:
            last_c = _last_compute_monotonic.get(driver_id, 0.0)
            if now_m - last_c < _ETA_COMPUTE_MIN_INTERVAL_SEC:
                inc_eta_changed_skipped(reason="throttle")
                return
            _last_compute_monotonic[driver_id] = now_m

        uc = GetDriverBookingsETAUseCase(
            eta_seconds_fn=get_eta_seconds_fn(),
            now_local_fn=now_local,
        )
        resp = uc.execute(
            driver_lat=driver_lat,
            driver_lon=driver_lon,
            bookings=bookings,
        )

        sig = _signature_from_response(resp)
        with _lock:
            prev = _last_eta_emitted_by_driver.get(driver_id)
            if not _significant_delta(prev, sig):
                inc_eta_changed_skipped(reason="no_significant_delta")
                return
            _last_eta_emitted_by_driver[driver_id] = sig

        payload = _serialize_eta_response(resp, bookings)
        emit_driver_event(driver_id, "eta_changed", payload)
        inc_eta_changed_emitted()
    except Exception:
        inc_eta_changed_skipped(reason="error")
        logger.exception(
            "maybe_emit_eta_changed_after_driver_location failed driver_id=%s",
            driver_id,
        )
