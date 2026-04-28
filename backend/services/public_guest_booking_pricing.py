"""Calcul du prix pour la réservation invitée (public), aligné sur le preview client."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any

from infrastructure.bookings.distance_duration import get_distance_duration_fn
from models import PricingProfile
from services.geo.geo_resolver import (
    geo_unit_id_from_pickup_admin_token,
    resolve_pickup_admin,
)
from services.geolocation.geocoding_interface import get_geocoding_service
from services.pricing.pricing_engine import compute_price
from shared.guest_booking_constants import GUEST_BOOKING_CUSTOMER_PLACEHOLDER
from shared.time_utils import api_scheduled_iso_to_naive_geneva

logger = logging.getLogger(__name__)

WEEKEND_START_WEEKDAY = 5
# Format "HH:MM" (5 caractères dont ':')
GUEST_PICKUP_TIME_HHMM_LEN = 5


def resolve_public_guest_pricing_company_id() -> int | None:
    """Entreprise dont le barème actif sert au calcul invité.

    - ``PUBLIC_GUEST_PRICING_COMPANY_ID`` : prioritaire en production.
    - Sinon : première entrée ``PricingProfile`` active (ordre ``company_id``), pratique en dev/tests.
    """
    raw = (os.getenv("PUBLIC_GUEST_PRICING_COMPANY_ID") or "").strip()
    if raw.isdigit():
        cid = int(raw)
        if cid > 0:
            return cid
    profile = (
        PricingProfile.query.filter_by(is_active=True)
        .order_by(PricingProfile.company_id.asc())
        .first()
    )
    if profile is None:
        return None
    return int(profile.company_id)


def _build_scheduled_iso_from_guest_fields(date_s: str, pickup_time_s: str) -> str:
    d = date_s.strip()
    t = pickup_time_s.strip()
    if not d or not t:
        return ""
    if len(t) == GUEST_PICKUP_TIME_HHMM_LEN and t[2] == ":":
        return f"{d}T{t}:00"
    if "T" in t:
        return f"{d}T{t}" if not t.startswith(d) else t
    return f"{d}T{t}"


def compute_public_guest_booking_price(  # noqa: PLR0911
    *,
    departure: str,
    destination: str,
    date: str,
    pickup_time: str,
    trip_type: str = "one_way",
) -> dict[str, Any]:
    """Calcule le prix comme ``POST /clients/me/bookings/preview`` (même moteur, même contexte).

    Returns:
        ``{"ok": True, "amount": float, "currency": str, ...}`` ou
        ``{"ok": False, "error": str, "error_message": str}``.
    """
    departure = departure.strip()
    destination = destination.strip()
    if not departure or not destination or not date.strip() or not pickup_time.strip():
        return {
            "ok": False,
            "error": "missing_fields",
            "error_message": "departure, destination, date et pickup_time sont requis.",
        }

    scheduled_iso = _build_scheduled_iso_from_guest_fields(date, pickup_time)
    scheduled_time = api_scheduled_iso_to_naive_geneva(scheduled_iso)
    if scheduled_time is None:
        return {
            "ok": False,
            "error": "invalid_schedule",
            "error_message": "Date ou heure de prise en charge invalides.",
        }

    geocoding = get_geocoding_service()
    pickup = geocoding.geocode_address(departure, country="CH")
    dropoff = geocoding.geocode_address(destination, country="CH")
    if not pickup or pickup.get("lat") is None or pickup.get("lon") is None:
        return {
            "ok": False,
            "error": "geocode_failed",
            "error_message": "Adresse de départ non géocodable.",
        }
    if not dropoff or dropoff.get("lat") is None or dropoff.get("lon") is None:
        return {
            "ok": False,
            "error": "geocode_failed",
            "error_message": "Adresse de destination non géocodable.",
        }

    pickup_lat = float(pickup["lat"])
    pickup_lng = float(pickup["lon"])
    dropoff_lat = float(dropoff["lat"])
    dropoff_lng = float(dropoff["lon"])

    distance_fn = get_distance_duration_fn()
    duration_seconds, distance_meters = distance_fn(departure, destination)

    pickup_admin = resolve_pickup_admin(
        lat=pickup_lat,
        lng=pickup_lng,
        pickup_zip=None,
        pickup_text=departure,
    )
    dropoff_admin = resolve_pickup_admin(
        lat=dropoff_lat,
        lng=dropoff_lng,
        pickup_zip=None,
        pickup_text=destination,
    )
    pickup_geo_unit_id = geo_unit_id_from_pickup_admin_token(
        str(pickup_admin.get("token") or "")
    )
    dropoff_geo_unit_id = geo_unit_id_from_pickup_admin_token(
        str(dropoff_admin.get("token") or "")
    )

    company_id = resolve_public_guest_pricing_company_id()
    if not company_id:
        return {
            "ok": False,
            "error": "pricing_not_configured",
            "error_message": "Aucun barème de prix invité configuré (entreprise / profil actif).",
        }

    profile = (
        PricingProfile.query.filter_by(company_id=company_id, is_active=True)
        .order_by(PricingProfile.created_at.desc())
        .first()
    )
    if not profile:
        return {
            "ok": False,
            "error": "pricing_not_configured",
            "error_message": "Aucun profil de pricing actif pour cette entreprise.",
        }

    version = profile.current_version
    if not version and profile.versions:
        version = sorted(profile.versions, key=lambda item: int(item.version), reverse=True)[0]
    if not version:
        return {
            "ok": False,
            "error": "pricing_not_configured",
            "error_message": "Aucune version de pricing active.",
        }

    is_round_trip = str(trip_type or "").strip().lower() == "round_trip"
    now_ref = (
        datetime.now(scheduled_time.tzinfo)
        if scheduled_time.tzinfo
        else datetime.now()
    )
    minutes_until = max(0, int((scheduled_time - now_ref).total_seconds() // 60))
    context: dict[str, Any] = {
        "is_weekend": scheduled_time.weekday() >= WEEKEND_START_WEEKDAY,
        "is_round_trip": is_round_trip,
        "pickup_local_time": scheduled_time.strftime("%H:%M"),
        "minutes_until_pickup": minutes_until,
        "distance_km": max(float(distance_meters or 0) / 1000.0, 0.0),
        "pickup_admin_token": pickup_admin.get("token"),
        "dropoff_admin_token": dropoff_admin.get("token"),
        "pickup_lat": pickup_lat,
        "pickup_lng": pickup_lng,
        "dropoff_lat": dropoff_lat,
        "dropoff_lng": dropoff_lng,
        "pickup_geo_unit_id": pickup_geo_unit_id,
        "dropoff_geo_unit_id": dropoff_geo_unit_id,
        "zones_count": (
            1 if pickup_admin.get("token") == dropoff_admin.get("token") else 2
        ),
        "requires_waiting": False,
    }

    booking_payload: dict[str, Any] = {
        "customer_name": GUEST_BOOKING_CUSTOMER_PLACEHOLDER,
        "pickup_location": departure,
        "dropoff_location": destination,
        "amount": 1.0,
    }

    try:
        amount, breakdown = compute_price(booking_payload, version, context)
    except Exception:
        logger.exception("compute_price invité (public guest booking)")
        return {
            "ok": False,
            "error": "pricing_compute_failed",
            "error_message": "Impossible de calculer le prix pour ce trajet.",
        }

    return {
        "ok": True,
        "amount": float(amount),
        "currency": str(profile.currency or "CHF"),
        "distance_meters": int(distance_meters or 0),
        "duration_seconds": int(duration_seconds or 0),
        "pickup_lat": float(pickup_lat),
        "pickup_lon": float(pickup_lng),
        "dropoff_lat": float(dropoff_lat),
        "dropoff_lon": float(dropoff_lng),
        "pickup_geo_unit_id": int(pickup_geo_unit_id) if pickup_geo_unit_id is not None else None,
        "dropoff_geo_unit_id": int(dropoff_geo_unit_id) if dropoff_geo_unit_id is not None else None,
        "pricing_profile_id": int(profile.id),
        "pricing_profile_version_id": int(version.id),
        "pricing_status": "confirmed",
        "breakdown": breakdown,
    }
