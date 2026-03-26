"""Invalidation du cache Redis pour GET /companies/me/reservations/summary."""

from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

logger = logging.getLogger(__name__)

# Longueur "YYYY-MM-DD" pour clés cache et préfixes de chaînes ISO.
_ISO_DATE_LEN = 10


def _day_str_from_booking(booking: Any) -> str | None:
    st = getattr(booking, "scheduled_time", None) or getattr(booking, "pickup_time", None)
    if st is None:
        return None
    if hasattr(st, "date") and callable(getattr(st, "date", None)):
        try:
            return st.date().isoformat()
        except Exception:
            pass
    if isinstance(st, str) and len(st) >= _ISO_DATE_LEN:
        return st[:_ISO_DATE_LEN]
    return None


def invalidate_reservations_summary_cache(company_id: int | None, day_str: str | None) -> None:
    """Supprime la clé summary:reservations:{company_id}:{YYYY-MM-DD} si Redis est disponible."""
    if company_id is None or not day_str:
        return
    day_clean = str(day_str).strip()[:_ISO_DATE_LEN]
    if len(day_clean) < _ISO_DATE_LEN:
        return
    try:
        from ext import redis_client
    except Exception:
        return
    if redis_client is None:
        return
    cache_key = f"summary:reservations:{int(company_id)}:{day_clean}"
    with suppress(Exception):
        redis_client.delete(cache_key)


def summary_day_for_booking(booking: Any) -> str | None:
    """Jour YYYY-MM-DD utilisé pour la clé Redis summary (scheduled_time / pickup_time)."""
    return _day_str_from_booking(booking)


def invalidate_summary_cache_for_booking(company_id: int | None, booking: Any) -> None:
    """Invalide le résumé pour le jour local de la réservation."""
    d = _day_str_from_booking(booking)
    if d:
        invalidate_reservations_summary_cache(company_id, d)


def invalidate_summary_cache_for_booking_after_day_change(
    company_id: int | None,
    booking: Any,
    previous_day_str: str | None,
) -> None:
    """Invalide le jour courant et, si la course a changé de jour, l'ancien jour aussi."""
    invalidate_summary_cache_for_booking(company_id, booking)
    new_day = _day_str_from_booking(booking)
    if (
        previous_day_str
        and new_day
        and previous_day_str != new_day
    ):
        invalidate_reservations_summary_cache(company_id, previous_day_str)
