from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from typing import Any

from ext import db
from models import Company, DispatchOffer, DispatchOfferStatus, DriverStatus, GeoUnit, ServiceArea
from services.geo.geo_resolver import canonical_reason, geo_chain, resolve_legacy_service_area_to_canton_codes


MATCH_SCORE = {
    "commune": 100,
    "zipcode": 90,
    "district": 70,
    "canton": 50,
    "country": 10,
}

THRESHOLD_WINDOWS = {100: 45, 70: 45, 50: 60, 10: 90}
URGENCY_WINDOW_SECONDS = 30
MAX_OFFERS_PER_BOOKING = 30
ONLINE_DRIVER_SECONDS = 120


@dataclass
class DispatchCandidate:
    company_id: int
    score: int
    reason: dict[str, Any]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def _mode_allowed(service_area: ServiceArea, pickup_match: bool, drop_match: bool) -> bool:
    mode = service_area.coverage_mode.value
    if mode in ("A_STRICT", "B_PICKUP_ONLY"):
        return pickup_match
    if mode == "C_INTRA_ONLY":
        return pickup_match and drop_match
    if mode == "D_NATIONAL":
        return True
    return False


def _best_company_candidate(
    *,
    company: Company,
    pickup_chain: list[GeoUnit],
    drop_chain: list[GeoUnit],
) -> DispatchCandidate | None:
    best_score = -1
    best_reason: dict[str, Any] | None = None
    pickup_map = {unit.id: unit for unit in pickup_chain}
    drop_ids = {unit.id for unit in drop_chain}

    for area in company.service_areas:
        if not area.is_active:
            continue
        pickup_match = area.geo_unit_id in pickup_map
        drop_match = area.geo_unit_id in drop_ids
        if not _mode_allowed(area, pickup_match, drop_match):
            continue

        if area.coverage_mode.value == "D_NATIONAL":
            unit_level = "country"
            base_score = MATCH_SCORE["country"]
            unit_id = area.geo_unit_id
        else:
            matched_unit = pickup_map.get(area.geo_unit_id)
            if not matched_unit:
                continue
            unit_level = matched_unit.type.value
            base_score = MATCH_SCORE.get(unit_level, 0)
            unit_id = matched_unit.id

        score = base_score + int(area.weight or 0)
        if score > best_score:
            best_score = score
            best_reason = canonical_reason(
                engine="dispatch_scoring_v1",
                threshold=0,
                pickup_level=unit_level,
                pickup_geo_unit_id=unit_id,
                coverage_mode=area.coverage_mode.value,
                weight=int(area.weight or 0),
            )

    if best_score <= 0 or not best_reason:
        return None
    return DispatchCandidate(company_id=company.id, score=best_score, reason=best_reason)


def compute_candidates(
    *,
    pickup_geo_unit: GeoUnit | None,
    drop_geo_unit: GeoUnit | None,
) -> list[DispatchCandidate]:
    pickup_chain = geo_chain(pickup_geo_unit)
    drop_chain = geo_chain(drop_geo_unit)
    companies = Company.query.filter(Company.dispatch_enabled.is_(True)).all()
    candidates: list[DispatchCandidate] = []

    for company in companies:
        best = _best_company_candidate(
            company=company,
            pickup_chain=pickup_chain,
            drop_chain=drop_chain,
        )
        if best:
            candidates.append(best)
            continue

        # Legacy fallback: parse free text service_area to cantons.
        legacy_codes = resolve_legacy_service_area_to_canton_codes(company.service_area)
        if not legacy_codes or not pickup_chain:
            continue
        pickup_canton = next((g for g in pickup_chain if g.type.value == "canton"), None)
        if pickup_canton and pickup_canton.code in legacy_codes:
            candidates.append(
                DispatchCandidate(
                    company_id=company.id,
                    score=MATCH_SCORE["canton"],
                    reason=canonical_reason(
                        engine="dispatch_scoring_v1",
                        threshold=0,
                        pickup_level="canton",
                        pickup_geo_unit_id=pickup_canton.id,
                        coverage_mode="LEGACY_FALLBACK",
                        weight=0,
                        legacy_fallback=True,
                    ),
                )
            )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    return candidates


def persist_offers_for_threshold(
    *,
    booking_id: int,
    candidates: list[DispatchCandidate],
    threshold: int,
) -> list[DispatchOffer]:
    existing_declined_ids = {
        row.company_id
        for row in DispatchOffer.query.filter(
            DispatchOffer.booking_id == booking_id,
            DispatchOffer.status == DispatchOfferStatus.DECLINED,
        ).all()
    }
    existing_company_ids = {
        row.company_id
        for row in DispatchOffer.query.filter(DispatchOffer.booking_id == booking_id).all()
    }

    accepted: list[DispatchOffer] = []
    for candidate in candidates:
        if candidate.score < threshold:
            continue
        if candidate.company_id in existing_declined_ids:
            continue
        if candidate.company_id in existing_company_ids:
            continue
        if len(accepted) >= MAX_OFFERS_PER_BOOKING:
            break

        expires_at = datetime.now(UTC) + timedelta(
            seconds=THRESHOLD_WINDOWS.get(threshold, 45)
        )
        reason = dict(candidate.reason)
        reason["threshold"] = threshold
        offer = DispatchOffer(
            booking_id=booking_id,
            company_id=candidate.company_id,
            status=DispatchOfferStatus.PROPOSED,
            score=candidate.score,
            reason_json=reason,
            expires_at=expires_at,
        )
        db.session.add(offer)
        accepted.append(offer)

    return accepted


def compute_urgency_override_candidates(
    *,
    pickup_lat: float | None,
    pickup_lon: float | None,
    radius_km: float = 3.0,
) -> list[dict[str, Any]]:
    if pickup_lat is None or pickup_lon is None:
        return []
    now = datetime.now(UTC)
    candidates: list[dict[str, Any]] = []
    rows = DriverStatus.query.filter(DriverStatus.last_update.isnot(None)).all()
    for row in rows:
        if row.latitude is None or row.longitude is None:
            continue
        if (now - row.last_update).total_seconds() > ONLINE_DRIVER_SECONDS:
            continue
        distance = haversine_km(pickup_lat, pickup_lon, row.latitude, row.longitude)
        if distance > radius_km:
            continue
        company_id = getattr(getattr(row, "driver", None), "company_id", None)
        if not company_id:
            continue
        candidates.append(
            {
                "company_id": company_id,
                "distance_km": round(distance, 3),
                "driver_last_seen_sec": int((now - row.last_update).total_seconds()),
            }
        )

    dedup: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        cid = candidate["company_id"]
        best = dedup.get(cid)
        if best is None or candidate["distance_km"] < best["distance_km"]:
            dedup[cid] = candidate
    return list(dedup.values())


def persist_urgency_offers(booking_id: int, candidates: list[dict[str, Any]]) -> list[DispatchOffer]:
    existing_company_ids = {
        row.company_id
        for row in DispatchOffer.query.filter(DispatchOffer.booking_id == booking_id).all()
    }
    created: list[DispatchOffer] = []
    for candidate in candidates:
        cid = candidate["company_id"]
        if cid in existing_company_ids:
            continue
        reason = {
            "engine": "dispatch_scoring_v1",
            "threshold": 999,
            "match": {"type": "urgency_proximity_override"},
            "flags": {"out_of_zone": True, "no_penalty_on_decline": True},
            "proximity": {
                "radius_km": 3,
                "driver_last_seen_sec": candidate["driver_last_seen_sec"],
                "approx_distance_km": candidate["distance_km"],
            },
        }
        offer = DispatchOffer(
            booking_id=booking_id,
            company_id=cid,
            status=DispatchOfferStatus.PROPOSED,
            score=999,
            reason_json=reason,
            expires_at=datetime.now(UTC) + timedelta(seconds=URGENCY_WINDOW_SECONDS),
        )
        db.session.add(offer)
        created.append(offer)
    return created
