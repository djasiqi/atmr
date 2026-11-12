"""Éligibilité du réseau LIRIE pour une demande institution.

La zone vient des ``ServiceArea`` structurées. Une zone absente n'ouvre pas
la Suisse. Aucun géocodage externe n'est appelé ici.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from application.companies.transport_capabilities import (
    declared_manual_wheelchair,
    declared_standard_assistance,
    specialized_escort_requested,
)
from models import Company, GeoUnit, GeoUnitType, TransportRequest
from models.service_area_pricing import ServiceArea
from models.vehicle import Vehicle
from services.geo.geo_resolver import (
    LEGACY_CANTON_MAP,
    geo_chain,
    normalize_text,
    resolve_geo_unit,
)

DISPATCH_NETWORK = "network"
DISPATCH_PARTNERS = "partners"
_COUNTRY = {"suisse", "switzerland", "schweiz", "france"}
_NPA = re.compile(r"\b(\d{4})\b")

# Besoins dont aucune preuve exploitable n'existe encore.
UNMODELED_NETWORK_NEEDS = (
    ("stretcher", "brancard"),
    ("oxygen", "oxygène"),
)


@dataclass(frozen=True, slots=True)
class NetworkSelection:
    companies: tuple[Company, ...] = ()
    error: str | None = None
    place_label: str | None = None


def resolve_dispatch_target(stored: str | None, *, has_preferences: bool) -> str:
    """NULL conserve les partenaires existants, sinon le réseau."""
    value = str(stored or "").strip().lower()
    if value in (DISPATCH_NETWORK, DISPATCH_PARTNERS):
        return value
    return DISPATCH_PARTNERS if has_preferences else DISPATCH_NETWORK


def unmodeled_network_needs(mobility: dict) -> list[str]:
    labels: list[str] = []
    for key, label in UNMODELED_NETWORK_NEEDS:
        if mobility.get(key):
            labels.append(label)
    if specialized_escort_requested(mobility):
        labels.append("accompagnement spécialisé")
    return labels


def place_label(pickup_location: str | None) -> str:
    unit = _local_unit(pickup_location)
    return unit.name if unit is not None else ""


def network_notice(*, place: str, when_label: str) -> str:
    """Annonce minimale : lieu et horaire, sans identité du patient."""
    where = place.strip() or "Lieu à confirmer"
    when = when_label.strip() or "horaire à confirmer"
    return f"Nouvelle demande de transport — {where}, {when}"


def area_covers(
    *,
    mode: str,
    area_geo_unit_id: int,
    pickup_ids: set[int],
    drop_ids: set[int],
    drop_known: bool,
    same_canton: bool,
) -> bool:
    """Même règles que le scoring, sans le repli texte libre."""
    pickup_match = area_geo_unit_id in pickup_ids
    drop_match = area_geo_unit_id in drop_ids
    if mode in ("A_STRICT", "B_PICKUP_ONLY"):
        return pickup_match
    if mode == "C_INTRA_ONLY":
        if not drop_known:
            return False
        if same_canton:
            return pickup_match and drop_match
        return pickup_match
    return mode == "D_NATIONAL"


def _local_unit(text: str | None) -> GeoUnit | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    while parts and normalize_text(parts[-1]) in _COUNTRY:
        parts.pop()
    for part in reversed(parts):
        cleaned = _NPA.sub("", part).strip(" ,")
        if len(cleaned) < 3:
            continue
        found = resolve_geo_unit(commune_text=cleaned, zip_code=None)
        if found.geo_unit_id:
            unit = GeoUnit.query.get(found.geo_unit_id)
            if unit is not None:
                return unit
    zip_match = _NPA.search(raw)
    if zip_match:
        found = resolve_geo_unit(commune_text=None, zip_code=zip_match.group(1))
        if found.geo_unit_id:
            unit = GeoUnit.query.get(found.geo_unit_id)
            if unit is not None:
                return unit
    for part in parts:
        token = normalize_text(_NPA.sub("", part))
        code = LEGACY_CANTON_MAP.get(token)
        if code:
            unit = GeoUnit.query.filter_by(type=GeoUnitType.CANTON, code=code).first()
            if unit is not None:
                return unit
        unit = GeoUnit.query.filter(
            GeoUnit.type == GeoUnitType.CANTON,
            GeoUnit.name.ilike(part.strip()),
        ).first()
        if unit is not None:
            return unit
    return None


def _canton_id(unit: GeoUnit | None) -> int | None:
    for item in geo_chain(unit):
        if item.type == GeoUnitType.CANTON:
            return int(item.id)
    return None


def select_network_companies(transport_request: TransportRequest) -> NetworkSelection:
    """Entreprises du catalogue qui couvrent la zone et le fauteuil personnel."""
    from application.institutions.eligible_carriers import (
        companies_for_marketplace_dispatch,
    )

    mobility = transport_request.get_mobility()
    missing = unmodeled_network_needs(mobility)
    if missing:
        listed = ", ".join(missing)
        return NetworkSelection(
            error=(
                "Le réseau LIRIE ne peut pas encore vérifier ces besoins : "
                f"{listed}. La demande n'a pas été envoyée."
            )
        )

    pickup = _local_unit(transport_request.pickup_location)
    if pickup is None:
        return NetworkSelection(
            error=(
                "La zone de prise en charge n'est pas reconnue dans le référentiel. "
                "La demande n'a pas été envoyée au réseau."
            )
        )
    drop = _local_unit(transport_request.dropoff_location)
    pickup_ids = {int(unit.id) for unit in geo_chain(pickup)}
    drop_ids = {int(unit.id) for unit in geo_chain(drop)} if drop is not None else set()
    pickup_canton = _canton_id(pickup)
    drop_canton = _canton_id(drop)
    same_canton = (
        pickup_canton is not None
        and drop_canton is not None
        and pickup_canton == drop_canton
    )

    catalog = companies_for_marketplace_dispatch(
        institution=transport_request.institution,
        require_official_acceptance=True,
    )
    if not catalog:
        return NetworkSelection(
            error=(
                "Aucun transporteur du réseau ne correspond actuellement à cette demande. "
                "Vous pouvez conserver votre brouillon ou réessayer ultérieurement."
            ),
            place_label=pickup.name,
        )

    company_ids = [int(company.id) for company in catalog]
    areas = ServiceArea.query.filter(
        ServiceArea.company_id.in_(company_ids),
        ServiceArea.is_active.is_(True),
    ).all()
    covered: set[int] = set()
    for area in areas:
        mode = getattr(area.coverage_mode, "value", area.coverage_mode)
        if area_covers(
            mode=str(mode),
            area_geo_unit_id=int(area.geo_unit_id),
            pickup_ids=pickup_ids,
            drop_ids=drop_ids,
            drop_known=drop is not None,
            same_canton=same_canton,
        ):
            covered.add(int(area.company_id))

    selected = [company for company in catalog if int(company.id) in covered]
    if mobility.get("wheelchair"):
        accessible = {
            int(company_id)
            for (company_id,) in Vehicle.query.filter(
                Vehicle.company_id.in_(list(covered) or [-1]),
                Vehicle.is_active.is_(True),
                Vehicle.wheelchair_accessible.is_(True),
            )
            .with_entities(Vehicle.company_id)
            .all()
        }
        selected = [company for company in selected if int(company.id) in accessible]

    if mobility.get("needs_assistance"):
        selected = [
            company for company in selected if declared_standard_assistance(company)
        ]
    if mobility.get("vehicle_wheelchair"):
        selected = [
            company for company in selected if declared_manual_wheelchair(company)
        ]

    if not selected:
        return NetworkSelection(
            error=(
                "Aucun transporteur du réseau ne correspond actuellement à cette demande. "
                "Vous pouvez conserver votre brouillon ou réessayer ultérieurement."
            ),
            place_label=pickup.name,
        )
    return NetworkSelection(companies=tuple(selected), place_label=pickup.name)
