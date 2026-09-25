"""Plafond PORTAL 7B.2 — MAX des tarifs transporteurs éligibles (pas de saisie client).

LIRIE n'invente aucun prix : chaque quote vient de la grille active du transporteur.
Le plafond = MAX(quotes calculables). Figé au BOOKING_CREATED.

Éligibilité « transporteur tarifairement éligible » (stable, documentée) :

  INCLUS si :
    - company.is_approved
    - dessert la zone de la mission (compute_candidates / géo pickup±drop)
      OU, si géo non résolue : entreprises approuvées
    - profil tarifaire actif en CHF
    - compute_price() → quote > 0

  EXCLUS (raisons tracées dans excluded[]) :
    - not_approved
    - hors zone (via compute_candidates)
    - no_active_pricing_profile
    - currency_not_chf
    - non_positive_quote / pricing_compute_failed

  NON pris en compte (volontairement) :
    - dispatch_enabled / mode MANUAL
      (MANUAL = pas d'auto-assign flotte interne ;
       une entreprise MANUAL approuvée reçoit toujours les missions LIRIE)
    - disponibilité instantanée d'un chauffeur
      (sinon le plafond varierait selon qui est occupé à T+0)

Si aucune quote : ValueError(portal_pricing_ceiling_unavailable) —
jamais de plafond inventé ni de 0 CHF silencieux.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

from models import Company, GeoUnit, PricingProfile
from services.dispatch.scoring_engine import compute_candidates
from services.geo.geo_resolver import (
    geo_unit_id_from_pickup_admin_token,
    resolve_pickup_admin,
)
from services.geolocation.maps import get_distance_duration
from services.pricing.offer_price_estimator import (
    DEFAULT_CURRENCY,
    _active_profile_version,
    _build_pricing_context,
)
from services.pricing.pricing_engine import compute_price

logger = logging.getLogger(__name__)

ERROR_PORTAL_PRICING_CEILING_UNAVAILABLE = "portal_pricing_ceiling_unavailable"


@dataclass(frozen=True)
class CarrierQuote:
    company_id: int
    company_name: str
    amount: float
    currency: str
    pricing_profile_id: int | None
    pricing_profile_version_id: int | None
    model: str | None = None


@dataclass(frozen=True)
class ExcludedCarrier:
    company_id: int
    company_name: str
    reason: str


@dataclass
class PortalCarrierCeiling:
    maximum_accepted_amount: float
    currency: str
    pricing_calculated_at: str
    eligible_carrier_count: int
    quoted_carrier_count: int
    quotes: list[CarrierQuote] = field(default_factory=list)
    excluded: list[ExcludedCarrier] = field(default_factory=list)
    distance_meters: int | None = None
    # Une demande PORTAL peut décrire une série (récurrence) : plafond = trajet × N.
    per_trip_maximum_accepted_amount: float | None = None
    series_occurrences: int = 1

    def to_evidence_dict(self) -> dict[str, Any]:
        return {
            "pricing_ceiling": {
                "currency": self.currency,
                "maximum_accepted_amount": self.maximum_accepted_amount,
                "per_trip_maximum_accepted_amount": (
                    self.per_trip_maximum_accepted_amount
                    if self.per_trip_maximum_accepted_amount is not None
                    else self.maximum_accepted_amount
                ),
                "series_occurrences": int(self.series_occurrences),
                "pricing_calculated_at": self.pricing_calculated_at,
                "eligible_carrier_count": self.eligible_carrier_count,
                "quoted_carrier_count": self.quoted_carrier_count,
                "distance_meters": self.distance_meters,
                "quotes": [asdict(q) for q in self.quotes],
                "excluded": [asdict(e) for e in self.excluded],
            }
        }

    def to_api_dict(self) -> dict[str, Any]:
        # Identités sans prix — disclosure pool (7B.5) ; quotes restent pour preuves serveur.
        from models import Company

        eligible_carriers: list[dict[str, Any]] = []
        seen: set[int] = set()
        for q in self.quotes:
            cid = int(q.company_id)
            if cid in seen:
                continue
            seen.add(cid)
            company = Company.query.get(cid)
            legal = (
                (
                    str(getattr(company, "legal_name", None) or "").strip()
                    if company is not None
                    else ""
                )
                or str(q.company_name or "").strip()
                or f"Entreprise #{cid}"
            )
            row: dict[str, Any] = {"company_id": cid, "legal_name": legal}
            if company is not None:
                uid = getattr(company, "uid_ide", None)
                if uid:
                    row["uid_ide"] = str(uid)
            eligible_carriers.append(row)
        return {
            "maximum_accepted_amount": self.maximum_accepted_amount,
            "per_trip_maximum_accepted_amount": (
                self.per_trip_maximum_accepted_amount
                if self.per_trip_maximum_accepted_amount is not None
                else self.maximum_accepted_amount
            ),
            "series_occurrences": int(self.series_occurrences),
            "currency": self.currency,
            "pricing_calculated_at": self.pricing_calculated_at,
            "eligible_carrier_count": self.eligible_carrier_count,
            "quoted_carrier_count": self.quoted_carrier_count,
            "distance_meters": self.distance_meters,
            "eligible_carriers": eligible_carriers,
            # Quotes individuelles non exposées au client dans l'API plafond.
        }


def _admin_token(
    *,
    lat: float | None,
    lon: float | None,
    text: str | None,
) -> str | None:
    admin = resolve_pickup_admin(
        lat=lat, lng=lon, pickup_zip=None, pickup_text=text
    )
    token = (admin.get("token") or "").strip()
    if token:
        return token
    cc = (admin.get("canton_code") or "").strip().upper()
    if cc:
        return f"canton:{cc}"
    return None


def _geo_unit_from_mission(
    *,
    lat: float | None,
    lon: float | None,
    text: str | None,
) -> GeoUnit | None:
    token = _admin_token(lat=lat, lon=lon, text=text)
    gid = geo_unit_id_from_pickup_admin_token(token)
    if gid is None:
        return None
    from extensions import db

    return db.session.get(GeoUnit, int(gid))


def _distance_meters(
    *,
    pickup_lat: float | None,
    pickup_lon: float | None,
    dropoff_lat: float | None,
    dropoff_lon: float | None,
    pickup_location: str | None,
    dropoff_location: str | None,
) -> int | None:
    try:
        if None not in (pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
            _, distance = get_distance_duration(
                (float(pickup_lat), float(pickup_lon)),
                (float(dropoff_lat), float(dropoff_lon)),
            )
            return int(distance)
        if pickup_location and dropoff_location:
            _, distance = get_distance_duration(pickup_location, dropoff_location)
            return int(distance)
    except Exception:
        logger.warning("Distance plafond PORTAL indisponible", exc_info=True)
    return None


def _eligible_companies(
    *,
    pickup_geo: GeoUnit | None,
    dropoff_geo: GeoUnit | None,
) -> tuple[list[Company], list[ExcludedCarrier]]:
    """Transporteurs pouvant recevoir une mission PORTAL (zone + approuvés).

    ``dispatch_enabled`` / mode MANUAL **n'excluent pas** :
    MANUAL coupe l'auto-assign flotte interne, pas la réception de missions LIRIE
    (même contrat que ``application.institutions.eligible_carriers``).

    Perf : on part des sociétés avec **profil tarifaire actif** (seules capables
    de produire une quote), puis on filtre par zone si la géo est résolue.
    """
    excluded: list[ExcludedCarrier] = []
    priced = (
        Company.query.join(
            PricingProfile, PricingProfile.company_id == Company.id
        )
        .filter(
            Company.is_approved.is_(True),
            PricingProfile.is_active.is_(True),
        )
        .distinct()
        .all()
    )

    if pickup_geo is not None:
        candidates = compute_candidates(
            pickup_geo_unit=pickup_geo,
            drop_geo_unit=dropoff_geo,
            require_dispatch_enabled=False,
        )
        company_ids = {c.company_id for c in candidates}
        if company_ids:
            companies = [c for c in priced if int(c.id) in company_ids]
            # Filet : grilles actives hors zone scoring mais service_area legacy
            # déjà couvertes via compute_candidates ; si aucun match zone,
            # garder les profils actifs (évite plafond bloqué à tort).
            if not companies:
                companies = list(priced)
        else:
            companies = list(priced)
    else:
        companies = list(priced)

    eligible: list[Company] = []
    for company in companies:
        name = str(getattr(company, "name", "") or f"#{company.id}")
        if not bool(getattr(company, "is_approved", False)):
            excluded.append(
                ExcludedCarrier(
                    company_id=int(company.id),
                    company_name=name,
                    reason="not_approved",
                )
            )
            continue
        eligible.append(company)
    return eligible, excluded


def compute_portal_carrier_ceiling(
    *,
    pickup_location: str | None,
    dropoff_location: str | None,
    pickup_lat: float | None = None,
    pickup_lon: float | None = None,
    dropoff_lat: float | None = None,
    dropoff_lon: float | None = None,
    scheduled_time: datetime | None = None,
    is_round_trip: bool = False,
    series_occurrences: int = 1,
) -> PortalCarrierCeiling:
    """Calcule le plafond = MAX(quotes) des grilles transporteurs éligibles.

    ``series_occurrences`` : une demande PORTAL récurrence = une réservation
    décrivant N passages → plafond demande = plafond trajet × N (comme l'indicatif).
    """
    pickup_geo = _geo_unit_from_mission(
        lat=pickup_lat, lon=pickup_lon, text=pickup_location
    )
    dropoff_geo = _geo_unit_from_mission(
        lat=dropoff_lat, lon=dropoff_lon, text=dropoff_location
    )
    eligible, excluded = _eligible_companies(
        pickup_geo=pickup_geo, dropoff_geo=dropoff_geo
    )

    distance_m = _distance_meters(
        pickup_lat=pickup_lat,
        pickup_lon=pickup_lon,
        dropoff_lat=dropoff_lat,
        dropoff_lon=dropoff_lon,
        pickup_location=pickup_location,
        dropoff_location=dropoff_location,
    )
    context = _build_pricing_context(
        scheduled_time=scheduled_time or datetime.now(UTC),
        distance_meters=distance_m,
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

    quotes: list[CarrierQuote] = []
    for company in eligible:
        name = str(getattr(company, "name", "") or f"#{company.id}")
        profile, version = _active_profile_version(int(company.id))
        if profile is None or version is None:
            excluded.append(
                ExcludedCarrier(
                    company_id=int(company.id),
                    company_name=name,
                    reason="no_active_pricing_profile",
                )
            )
            continue
        currency = str(getattr(profile, "currency", None) or DEFAULT_CURRENCY).upper()
        if currency != "CHF":
            excluded.append(
                ExcludedCarrier(
                    company_id=int(company.id),
                    company_name=name,
                    reason="currency_not_chf",
                )
            )
            continue
        try:
            # Source de vérité unique : même compute_price que les devis transporteur.
            # Aucune formule flat/distance/zone_count ici.
            amount, breakdown = compute_price(booking_payload, version, context)
            amount_dec = (
                amount
                if isinstance(amount, Decimal)
                else Decimal(str(amount))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            if amount_dec <= 0:
                excluded.append(
                    ExcludedCarrier(
                        company_id=int(company.id),
                        company_name=name,
                        reason="non_positive_quote",
                    )
                )
                continue
            model = None
            if isinstance(breakdown, dict):
                model = breakdown.get("model")
            quotes.append(
                CarrierQuote(
                    company_id=int(company.id),
                    company_name=name,
                    amount=float(amount_dec),
                    currency="CHF",
                    pricing_profile_id=int(profile.id),
                    pricing_profile_version_id=int(version.id),
                    model=str(model) if model else None,
                )
            )
        except Exception:
            logger.info(
                "Quote plafond PORTAL impossible company=%s",
                company.id,
                exc_info=True,
            )
            excluded.append(
                ExcludedCarrier(
                    company_id=int(company.id),
                    company_name=name,
                    reason="pricing_compute_failed",
                )
            )

    if not quotes:
        raise ValueError(ERROR_PORTAL_PRICING_CEILING_UNAVAILABLE)

    per_trip = max(
        Decimal(str(q.amount)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        for q in quotes
    )
    try:
        occ = max(1, min(52, int(series_occurrences or 1)))
    except (TypeError, ValueError):
        occ = 1
    ceiling_dec = (per_trip * Decimal(occ)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
    return PortalCarrierCeiling(
        maximum_accepted_amount=float(ceiling_dec),
        currency="CHF",
        pricing_calculated_at=datetime.now(UTC).isoformat(),
        eligible_carrier_count=len(eligible),
        quoted_carrier_count=len(quotes),
        quotes=sorted(quotes, key=lambda q: q.amount, reverse=True),
        excluded=excluded,
        distance_meters=distance_m,
        per_trip_maximum_accepted_amount=float(per_trip),
        series_occurrences=occ,
    )


def estimate_portal_carrier_offer_amount(
    booking: Any,
    company_id: int,
) -> float | None:
    """Quote grille du transporteur pour une mission PORTAL (jamais ``booking.amount``).

    Utilisé à l'acceptation (fallback ``offered_amount``) et à l'affichage dashboard
    entreprise. Ne révèle **pas** le plafond client.
    """
    profile, version = _active_profile_version(int(company_id))
    if profile is None or version is None:
        return None
    currency = str(getattr(profile, "currency", None) or DEFAULT_CURRENCY).upper()
    if currency != "CHF":
        return None

    pickup_location = getattr(booking, "pickup_location", None)
    dropoff_location = getattr(booking, "dropoff_location", None)
    pickup_lat = getattr(booking, "pickup_lat", None)
    pickup_lon = getattr(booking, "pickup_lon", None)
    dropoff_lat = getattr(booking, "dropoff_lat", None)
    dropoff_lon = getattr(booking, "dropoff_lon", None)
    scheduled_time = getattr(booking, "scheduled_time", None)
    is_round_trip = bool(getattr(booking, "is_round_trip", False))

    distance_m = _distance_meters(
        pickup_lat=float(pickup_lat) if pickup_lat is not None else None,
        pickup_lon=float(pickup_lon) if pickup_lon is not None else None,
        dropoff_lat=float(dropoff_lat) if dropoff_lat is not None else None,
        dropoff_lon=float(dropoff_lon) if dropoff_lon is not None else None,
        pickup_location=pickup_location,
        dropoff_location=dropoff_location,
    )
    context = _build_pricing_context(
        scheduled_time=scheduled_time or datetime.now(UTC),
        distance_meters=distance_m,
        is_round_trip=is_round_trip,
        pickup_lat=float(pickup_lat) if pickup_lat is not None else None,
        pickup_lon=float(pickup_lon) if pickup_lon is not None else None,
        dropoff_lat=float(dropoff_lat) if dropoff_lat is not None else None,
        dropoff_lon=float(dropoff_lon) if dropoff_lon is not None else None,
        pickup_location=pickup_location,
        dropoff_location=dropoff_location,
    )
    booking_payload = {
        "pickup_location": pickup_location,
        "dropoff_location": dropoff_location,
        "is_round_trip": is_round_trip,
    }
    try:
        amount, _breakdown = compute_price(booking_payload, version, context)
        amount_dec = (
            amount
            if isinstance(amount, Decimal)
            else Decimal(str(amount or 0))
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if amount_dec <= 0:
            return None
        return float(amount_dec)
    except Exception:
        logger.warning(
            "estimate_portal_carrier_offer_amount échoué booking_id=%s company_id=%s",
            getattr(booking, "id", None),
            company_id,
            exc_info=True,
        )
        return None
