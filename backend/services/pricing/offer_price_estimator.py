"""Estimation tarifaire des offres de demande institution.

Règle métier (payeur effectif) :
- ``billing_intent`` effectif = ``institution`` :
  tarif préférentiel du client institution si présent, sinon profil tarifaire actif.
- ``billing_intent`` effectif ≠ ``institution`` (patient, curateur, assurance, etc.) :
  jamais de tarif préférentiel institution — uniquement le profil tarifaire actif.

Ce module alimente l'estimation affichée sur l'offre et le gel tarifaire à l'acceptation.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from typing import Any

from flask import has_app_context

from models import PricingProfile
from services.geo.geo_resolver import (
    geo_unit_id_from_pickup_admin_token,
    resolve_pickup_admin,
)
from services.geolocation.maps import get_distance_duration
from services.pricing.pricing_engine import compute_price

logger = logging.getLogger(__name__)

WEEKEND_START_INDEX = 5
DEFAULT_MIN_AMOUNT = 0.5
DEFAULT_CURRENCY = "CHF"

SOURCE_PREFERENTIAL = "preferential"
SOURCE_COMPANY_PROFILE = "company_profile"
SOURCE_DEFAULT = "default"
SOURCE_MIXED = "mixed"


def institution_preferential_applies(effective_billing_intent: str | None) -> bool:
    """True si le payeur effectif autorise le tarif préférentiel institution."""
    return (effective_billing_intent or "patient").lower() == "institution"


def effective_preferential_rate(
    preferential_rate: Any,
    effective_billing_intent: str | None,
) -> Any:
    """Neutralise le tarif préférentiel institution si le payeur n'est pas institution."""
    if not institution_preferential_applies(effective_billing_intent):
        return None
    return preferential_rate


def _to_positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _active_profile_version(company_id: int):
    """Retourne (profil actif, version active) ou (None, None)."""
    profile = (
        PricingProfile.query.filter_by(company_id=company_id, is_active=True)
        .order_by(PricingProfile.created_at.desc())
        .first()
    )
    if not profile:
        return None, None
    version = profile.current_version
    if not version and profile.versions:
        version = sorted(
            profile.versions, key=lambda item: int(item.version), reverse=True
        )[0]
    return profile, version


def _compute_distance_meters(
    pickup_lat: float | None,
    pickup_lon: float | None,
    dropoff_lat: float | None,
    dropoff_lon: float | None,
    pickup_location: str | None,
    dropoff_location: str | None,
) -> int | None:
    try:
        if (
            pickup_lat is not None
            and pickup_lon is not None
            and dropoff_lat is not None
            and dropoff_lon is not None
        ):
            _, distance = get_distance_duration(
                (float(pickup_lat), float(pickup_lon)),
                (float(dropoff_lat), float(dropoff_lon)),
            )
            return int(distance)
        if pickup_location and dropoff_location:
            _, distance = get_distance_duration(pickup_location, dropoff_location)
            return int(distance)
    except Exception:
        logger.warning(
            "Calcul de distance impossible pour l'estimation tarifaire", exc_info=True
        )
    return None


def _build_pricing_context(
    *,
    scheduled_time: datetime | None,
    distance_meters: int | None,
    is_round_trip: bool,
    pickup_lat: float | None,
    pickup_lon: float | None,
    dropoff_lat: float | None,
    dropoff_lon: float | None,
    pickup_location: str | None,
    dropoff_location: str | None,
) -> dict[str, Any]:
    sched = scheduled_time or datetime.now()
    now_ref = datetime.now(sched.tzinfo) if sched.tzinfo else datetime.now()
    minutes_until = max(0, int((sched - now_ref).total_seconds() // 60))

    pickup_admin = resolve_pickup_admin(
        lat=pickup_lat, lng=pickup_lon, pickup_zip=None, pickup_text=pickup_location
    )
    dropoff_admin = resolve_pickup_admin(
        lat=dropoff_lat, lng=dropoff_lon, pickup_zip=None, pickup_text=dropoff_location
    )
    pickup_token = pickup_admin.get("token")
    dropoff_token = dropoff_admin.get("token")

    return {
        "is_weekend": sched.weekday() >= WEEKEND_START_INDEX,
        "is_round_trip": bool(is_round_trip),
        "pickup_local_time": sched.strftime("%H:%M"),
        "minutes_until_pickup": minutes_until,
        "distance_km": max(float(distance_meters or 0) / 1000.0, 0.0),
        "pickup_admin_token": pickup_token,
        "dropoff_admin_token": dropoff_token,
        "pickup_lat": pickup_lat,
        "pickup_lng": pickup_lon,
        "dropoff_lat": dropoff_lat,
        "dropoff_lng": dropoff_lon,
        "pickup_geo_unit_id": geo_unit_id_from_pickup_admin_token(
            str(pickup_token or "")
        ),
        "dropoff_geo_unit_id": geo_unit_id_from_pickup_admin_token(
            str(dropoff_token or "")
        ),
        "zones_count": 1 if pickup_token == dropoff_token else 2,
        "requires_waiting": False,
    }


def _aggregate_price_sources(sources: set[str]) -> str:
    if len(sources) > 1:
        return SOURCE_MIXED
    if SOURCE_COMPANY_PROFILE in sources:
        return SOURCE_COMPANY_PROFILE
    if SOURCE_PREFERENTIAL in sources:
        return SOURCE_PREFERENTIAL
    return SOURCE_DEFAULT


def resolve_institution_price(
    *,
    company_id: int | None,
    effective_billing_intent: str | None = "patient",
    preferential_rate: Any = None,
    pickup_location: str | None = None,
    dropoff_location: str | None = None,
    pickup_lat: float | None = None,
    pickup_lon: float | None = None,
    dropoff_lat: float | None = None,
    dropoff_lon: float | None = None,
    scheduled_time: datetime | None = None,
    is_round_trip: bool = False,
) -> dict[str, Any]:
    """Résout le tarif d'un trajet institution selon le payeur effectif.

    Retourne un dict :
        {amount, currency, source, pricing_profile_id,
         pricing_profile_version_id, breakdown}
    source ∈ {preferential, company_profile, default, mixed}.
    """
    preferential = _to_positive(
        effective_preferential_rate(preferential_rate, effective_billing_intent)
    )
    if preferential is not None:
        return {
            "amount": preferential,
            "currency": DEFAULT_CURRENCY,
            "source": SOURCE_PREFERENTIAL,
            "pricing_profile_id": None,
            "pricing_profile_version_id": None,
            "breakdown": None,
        }

    if has_app_context() and company_id:
        profile, version = _active_profile_version(int(company_id))
        if profile and version:
            distance_meters = _compute_distance_meters(
                pickup_lat,
                pickup_lon,
                dropoff_lat,
                dropoff_lon,
                pickup_location,
                dropoff_location,
            )
            context = _build_pricing_context(
                scheduled_time=scheduled_time,
                distance_meters=distance_meters,
                is_round_trip=is_round_trip,
                pickup_lat=pickup_lat,
                pickup_lon=pickup_lon,
                dropoff_lat=dropoff_lat,
                dropoff_lon=dropoff_lon,
                pickup_location=pickup_location,
                dropoff_location=dropoff_location,
            )
            booking_payload = {
                "pickup_location": pickup_location,
                "dropoff_location": dropoff_location,
                "is_round_trip": bool(is_round_trip),
            }
            try:
                amount, breakdown = compute_price(booking_payload, version, context)
                return {
                    "amount": float(amount),
                    "currency": getattr(profile, "currency", None) or DEFAULT_CURRENCY,
                    "source": SOURCE_COMPANY_PROFILE,
                    "pricing_profile_id": profile.id,
                    "pricing_profile_version_id": version.id,
                    "breakdown": breakdown,
                }
            except Exception:
                logger.exception(
                    "Calcul du profil tarifaire échoué (company=%s)", company_id
                )

    return {
        "amount": DEFAULT_MIN_AMOUNT,
        "currency": DEFAULT_CURRENCY,
        "source": SOURCE_DEFAULT,
        "pricing_profile_id": None,
        "pricing_profile_version_id": None,
        "breakdown": None,
    }


def _normalize_institution_name(value: str | None) -> str:
    """Normalise un nom d'institution pour matching tolérant (accents, casse)."""
    if not value:
        return ""
    normalized = unicodedata.normalize("NFKD", str(value))
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = (
        normalized.lower().replace("'", "'").replace("`", "'").replace("´", "'")
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _resolve_institution_client_readonly(request: Any, company_id: int | None):
    """Recherche (sans création) le client institution pour récupérer le tarif."""
    from models.client import Client

    institution = getattr(request, "institution", None)
    if not institution or not company_id:
        return None
    client = Client.query.filter(
        Client.company_id == company_id,
        Client.is_institution.is_(True),
        Client.linked_institution_id == institution.id,
    ).first()
    if client:
        return client

    target = _normalize_institution_name(getattr(institution, "name", None))
    if not target:
        return None
    candidates = Client.query.filter(
        Client.company_id == company_id,
        Client.is_institution.is_(True),
    ).all()
    for candidate in candidates:
        candidate_name = _normalize_institution_name(
            getattr(candidate, "institution_name", None)
        )
        if candidate_name and (
            candidate_name == target
            or candidate_name in target
            or target in candidate_name
        ):
            return candidate
    return None


def estimate_offer_price(offer: Any) -> dict[str, Any] | None:
    """Estime le tarif total d'une offre (somme des legs si multi-stop).

    Lecture seule — destiné à l'affichage. Retourne ``None`` en cas d'échec.
    """
    try:
        from services.billing.destination_billing_resolver import (
            effective_billing_for_leg,
        )

        request = getattr(offer, "transport_request", None)
        if request is None:
            return None
        company_id = getattr(offer, "company_id", None)

        institution_client = _resolve_institution_client_readonly(request, company_id)
        preferential_rate = (
            getattr(institution_client, "preferential_rate", None)
            if institution_client
            else None
        )

        scheduled_time = getattr(request, "scheduled_time", None)
        is_round_trip = bool(getattr(request, "is_round_trip", False))
        primary_intent = (getattr(request, "billing_intent", None) or "patient").lower()

        legs = sorted(
            getattr(request, "legs", None) or [],
            key=lambda leg: (getattr(leg, "sequence_index", 0) or 0),
        )

        total = 0.0
        sources: set[str] = set()
        currency = DEFAULT_CURRENCY

        if legs:
            leg_targets = [
                (
                    effective_billing_for_leg(leg, request),
                    {
                        "pickup_location": leg.pickup_location,
                        "dropoff_location": leg.dropoff_location,
                        "pickup_lat": float(leg.pickup_lat) if leg.pickup_lat else None,
                        "pickup_lon": float(leg.pickup_lng) if leg.pickup_lng else None,
                        "dropoff_lat": float(leg.dropoff_lat)
                        if leg.dropoff_lat
                        else None,
                        "dropoff_lon": float(leg.dropoff_lng)
                        if leg.dropoff_lng
                        else None,
                    },
                )
                for leg in legs
            ]
        else:
            leg_targets = [
                (
                    primary_intent,
                    {
                        "pickup_location": request.pickup_location,
                        "dropoff_location": request.dropoff_location,
                        "pickup_lat": float(request.pickup_lat)
                        if request.pickup_lat
                        else None,
                        "pickup_lon": float(request.pickup_lng)
                        if request.pickup_lng
                        else None,
                        "dropoff_lat": float(request.dropoff_lat)
                        if request.dropoff_lat
                        else None,
                        "dropoff_lon": float(request.dropoff_lng)
                        if request.dropoff_lng
                        else None,
                    },
                )
            ]

        for effective_intent, target in leg_targets:
            result = resolve_institution_price(
                company_id=company_id,
                effective_billing_intent=effective_intent,
                preferential_rate=preferential_rate,
                scheduled_time=scheduled_time,
                is_round_trip=is_round_trip,
                **target,
            )
            total += float(result["amount"])
            sources.add(result["source"])
            currency = result["currency"]

        return {
            "amount": round(total, 2),
            "currency": currency,
            "source": _aggregate_price_sources(sources),
        }
    except Exception:
        logger.warning(
            "Estimation tarifaire impossible pour l'offre %s",
            getattr(offer, "id", None),
            exc_info=True,
        )
        return None
