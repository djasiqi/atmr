"""Public platform statistics endpoint — no auth required, cached."""
from __future__ import annotations

import time
from threading import Lock

from flask_restx import Namespace, Resource
from sqlalchemy import func

from ext import db
from models.booking import Booking
from models.company import Company
from models.driver import Driver
from models.enums import BookingStatus

public_stats_ns = Namespace("public_stats", description="Statistiques publiques de la plateforme")

_cache: dict | None = None
_cache_ts: float = 0.0
_cache_lock = Lock()
CACHE_TTL = 300  # 5 minutes


def _fetch_stats() -> dict:
    completed = db.session.query(func.count(Booking.id)).filter(
        Booking.status == BookingStatus.COMPLETED
    ).scalar() or 0

    active_companies = db.session.query(func.count(Company.id)).filter(
        Company.is_approved.is_(True)
    ).scalar() or 0

    active_drivers = db.session.query(func.count(Driver.id)).filter(
        Driver.is_active.is_(True)
    ).scalar() or 0

    cities_served = db.session.query(
        func.count(func.distinct(Company.domicile_city))
    ).filter(
        Company.is_approved.is_(True),
        Company.domicile_city.isnot(None),
        Company.domicile_city != "",
    ).scalar() or 0

    return {
        "completedBookings": completed,
        "activeCompanies": active_companies,
        "activeDrivers": active_drivers,
        "citiesServed": cities_served,
    }


@public_stats_ns.route("")
class PlatformStats(Resource):
    def get(self):
        """Statistiques publiques de la plateforme (cache 5 min)."""
        global _cache, _cache_ts

        now = time.monotonic()
        if _cache is not None and (now - _cache_ts) < CACHE_TTL:
            return _cache

        with _cache_lock:
            if _cache is not None and (time.monotonic() - _cache_ts) < CACHE_TTL:
                return _cache
            stats = _fetch_stats()
            _cache = stats
            _cache_ts = time.monotonic()
            return stats
