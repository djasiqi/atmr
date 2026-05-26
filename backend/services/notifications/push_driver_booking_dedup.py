"""Dedup court-terme push mission assignée (évite double notif BookingAssigned + DriverNewBooking)."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_DEDUP_NS = os.getenv("DRIVER_PUSH_DEDUP_REDIS_NS", "atmr:driver_push:dedup")
_DEFAULT_TTL_SEC = int(os.getenv("DRIVER_PUSH_DEDUP_TTL_SEC", "45"))
_ENABLED = os.getenv("DRIVER_PUSH_DEDUP_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)


def _redis():
    from ext import redis_client

    return redis_client


def claim_driver_booking_push(
    driver_id: int,
    booking_id: int,
    *,
    ttl_sec: int | None = None,
) -> bool:
    """Retourne True si la push peut être envoyée (premier claim), False si doublon récent.

    Fail-open si Redis indisponible ou dedup désactivé.
    """
    if not _ENABLED:
        return True
    if driver_id <= 0 or booking_id <= 0:
        return True

    rc = _redis()
    if not rc:
        return True

    ttl = max(30, min(60, ttl_sec if ttl_sec is not None else _DEFAULT_TTL_SEC))
    key = f"{_DEDUP_NS}:{driver_id}:{booking_id}"
    try:
        ok = rc.set(key, "1", nx=True, ex=ttl)
        return bool(ok)
    except Exception as exc:
        logger.debug("claim_driver_booking_push fail-open: %s", exc)
        return True
