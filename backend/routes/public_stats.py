"""Public platform statistics endpoint — no auth required, cached."""

from __future__ import annotations

import json
import re
import time
import unicodedata
from threading import Lock
from urllib.parse import unquote

from flask_restx import Namespace, Resource
from sqlalchemy import func

from ext import db
from models.booking import Booking
from models.company import Company
from models.driver import Driver
from models.enums import BookingStatus

# Aligné sur routes.geocode — évite un import lourd au chargement du module
_SWISS_CANTON_CODES = frozenset(
    {
        "AG",
        "AI",
        "AR",
        "BE",
        "BL",
        "BS",
        "FR",
        "GE",
        "GL",
        "GR",
        "JU",
        "LU",
        "NE",
        "NW",
        "OW",
        "SG",
        "SH",
        "SO",
        "SZ",
        "TG",
        "TI",
        "UR",
        "VD",
        "VS",
        "ZG",
        "ZH",
    }
)

_CANTON_TOKEN_RE = re.compile(r"^canton:([A-Za-z0-9]+)$", re.IGNORECASE)
_CANTON_NAME_TOKEN_RE = re.compile(r"^canton_name:(.+)$", re.IGNORECASE)


def _norm_canton_name_lookup(value: str) -> str:
    return (
        unicodedata.normalize("NFD", value or "")
        .encode("ascii", "ignore")
        .decode("ascii")
        .strip()
        .lower()
    )


_NAME_TO_CODE_CACHE: dict[str, str] | None = None


def _swiss_canton_name_to_code() -> dict[str, str]:
    """Lazy import des libellés canton → code (réutilise la référence geo)."""
    global _NAME_TO_CODE_CACHE
    if _NAME_TO_CODE_CACHE is None:
        from routes.geocode import SWISS_CANTON_NAME_TO_CODE

        _NAME_TO_CODE_CACHE = SWISS_CANTON_NAME_TO_CODE
    return _NAME_TO_CODE_CACHE


def _codes_from_token_string(token: str, name_to_code: dict[str, str]) -> set[str]:
    out: set[str] = set()
    if not token:
        return out
    m = _CANTON_TOKEN_RE.match(token.strip())
    if m:
        code = m.group(1).upper()
        if code in _SWISS_CANTON_CODES:
            out.add(code)
        return out
    m = _CANTON_NAME_TOKEN_RE.match(token.strip())
    if m:
        name = unquote(m.group(1))
        code = name_to_code.get(_norm_canton_name_lookup(name))
        if code:
            out.add(code)
    return out


def _cantons_from_service_area(
    raw: str | None, name_to_code: dict[str, str]
) -> set[str]:
    """Extrait les codes canton (ISO) depuis service_area JSON V1 ou legacy CSV."""
    out: set[str] = set()
    s = (raw or "").strip()
    if not s:
        return out
    parsed_as_json = False
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            parsed_as_json = True
            toks = data.get("tokens")
            if isinstance(toks, list):
                for t in toks:
                    out.update(_codes_from_token_string(str(t).strip(), name_to_code))
    except json.JSONDecodeError:
        pass
    if not parsed_as_json:
        for part in s.split(","):
            out.update(_codes_from_token_string(part.strip(), name_to_code))
    return out


public_stats_ns = Namespace(
    "public_stats", description="Statistiques publiques de la plateforme"
)

_cache: dict | None = None
_cache_ts: float = 0.0
_cache_lock = Lock()
CACHE_TTL = 300  # 5 minutes


def _fetch_stats() -> dict:
    completed = (
        db.session.query(func.count(Booking.id))
        .filter(Booking.status == BookingStatus.COMPLETED)
        .scalar()
        or 0
    )

    active_companies = (
        db.session.query(func.count(Company.id))
        .filter(Company.is_approved.is_(True))
        .scalar()
        or 0
    )

    active_drivers = (
        db.session.query(func.count(Driver.id))
        .filter(Driver.is_active.is_(True))
        .scalar()
        or 0
    )

    name_to_code = _swiss_canton_name_to_code()
    cantons_served: set[str] = set()
    for (service_area,) in (
        db.session.query(Company.service_area)
        .filter(
            Company.is_approved.is_(True),
            Company.service_area.isnot(None),
            Company.service_area != "",
        )
        .all()
    ):
        cantons_served.update(_cantons_from_service_area(service_area, name_to_code))

    return {
        "completedBookings": completed,
        "activeCompanies": active_companies,
        "activeDrivers": active_drivers,
        "cantonsServed": len(cantons_served),
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
