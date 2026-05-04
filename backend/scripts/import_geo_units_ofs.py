"""Import OFS-like Swiss geographic hierarchy into geo_unit table.

Expected CSV columns:
- type (country|canton|district|commune|zipcode)
- code
- name
- parent_type (optional)
- parent_code (optional)

Optional geometry enrichment (PostGIS):
- fetches commune polygons from geo.admin by commune BFS code
- stores geometry in geo_unit.geom as MultiPolygon SRID 4326
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import requests
from sqlalchemy import text

from ext import db
from models import GeoUnit, PlatformZoneMembership
from models.enums import GeoUnitType

GEOADMIN_BASE_URL = os.getenv("GEOADMIN_BASE_URL", "https://api3.geo.admin.ch").rstrip(
    "/"
)
GEOMETRY_TIMEOUT_SECONDS = float(os.getenv("GEO_GEOMETRY_FETCH_TIMEOUT", "8"))


DEFAULT_ROWS = [
    {
        "type": "country",
        "code": "CH",
        "name": "Suisse",
        "parent_type": "",
        "parent_code": "",
    },
    {
        "type": "canton",
        "code": "GE",
        "name": "Genève",
        "parent_type": "country",
        "parent_code": "CH",
    },
    {
        "type": "canton",
        "code": "VD",
        "name": "Vaud",
        "parent_type": "country",
        "parent_code": "CH",
    },
    {
        "type": "canton",
        "code": "VS",
        "name": "Valais",
        "parent_type": "country",
        "parent_code": "CH",
    },
]


def load_rows(csv_path: str | None) -> list[dict[str, str]]:
    if not csv_path:
        return DEFAULT_ROWS
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def import_geo_units(csv_path: str | None = None) -> dict[str, int]:
    rows = load_rows(csv_path)
    created = 0
    updated = 0

    # Pass 1: create/update nodes without parent linkage.
    for row in rows:
        type_value = row.get("type", "").strip()
        code = row.get("code", "").strip()
        name = row.get("name", "").strip()
        if not type_value or not code or not name:
            continue
        node = GeoUnit.query.filter_by(type=GeoUnitType(type_value), code=code).first()
        if not node:
            node = GeoUnit(type=GeoUnitType(type_value), code=code, name=name)
            db.session.add(node)
            created += 1
        elif node.name != name:
            node.name = name
            updated += 1

    db.session.flush()

    # Pass 2: parent linkage.
    for row in rows:
        type_value = row.get("type", "").strip()
        code = row.get("code", "").strip()
        parent_type = row.get("parent_type", "").strip()
        parent_code = row.get("parent_code", "").strip()
        if not type_value or not code:
            continue
        node = GeoUnit.query.filter_by(type=GeoUnitType(type_value), code=code).first()
        if not node:
            continue
        if not parent_type or not parent_code:
            node.parent_id = None
            continue
        parent = GeoUnit.query.filter_by(
            type=GeoUnitType(parent_type), code=parent_code
        ).first()
        node.parent_id = parent.id if parent else None

    db.session.commit()
    return {"created": created, "updated": updated, "total_rows": len(rows)}


def _fetch_commune_feature(commune_code: str) -> dict[str, Any] | None:
    code = str(commune_code or "").strip()
    if not code.isdigit():
        return None
    endpoint = (
        f"{GEOADMIN_BASE_URL}/rest/services/api/MapServer/"
        f"ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill/{code}"
    )
    params = {"geometryFormat": "geojson", "sr": 4326}
    try:
        response = requests.get(
            endpoint, params=params, timeout=GEOMETRY_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return None
    feature = payload.get("feature") if isinstance(payload, dict) else None
    if not isinstance(feature, dict):
        return None
    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        return None
    return feature


def _fetch_commune_geometry_geojson(commune_code: str) -> dict[str, Any] | None:
    feature = _fetch_commune_feature(commune_code)
    geometry = feature.get("geometry") if isinstance(feature, dict) else None
    return geometry if isinstance(geometry, dict) else None


def _to_multipolygon_geojson(geometry: dict[str, Any]) -> dict[str, Any] | None:
    gtype = str(geometry.get("type") or "")
    coords = geometry.get("coordinates")
    if gtype == "MultiPolygon" and isinstance(coords, list):
        return {"type": "MultiPolygon", "coordinates": coords}
    if gtype == "Polygon" and isinstance(coords, list):
        return {"type": "MultiPolygon", "coordinates": [coords]}
    return None


def populate_geo_unit_geom_from_geoadmin(
    *, dry_run: bool = False, limit: int | None = None
) -> dict[str, int]:
    communes_query = GeoUnit.query.filter(GeoUnit.type == GeoUnitType.COMMUNE).order_by(
        GeoUnit.code.asc()
    )
    if limit and limit > 0:
        communes = communes_query.limit(limit).all()
    else:
        communes = communes_query.all()

    checked = 0
    updated = 0
    missing = 0
    failed = 0
    for commune in communes:
        checked += 1
        geometry = _fetch_commune_geometry_geojson(str(commune.code))
        if not geometry:
            missing += 1
            continue
        multi = _to_multipolygon_geojson(geometry)
        if not multi:
            failed += 1
            continue
        if dry_run:
            updated += 1
            continue
        try:
            with db.session.begin_nested():
                db.session.execute(
                    text(
                        """
                        UPDATE geo_unit
                        SET geom = ST_SetSRID(ST_GeomFromGeoJSON(:geojson), 4326)
                        WHERE id = :unit_id
                        """
                    ),
                    {
                        "geojson": json.dumps(
                            multi, ensure_ascii=False, separators=(",", ":")
                        ),
                        "unit_id": commune.id,
                    },
                )
            updated += 1
        except Exception:
            failed += 1

    if not dry_run:
        db.session.commit()
    return {
        "checked": checked,
        "updated": updated,
        "missing_geometry": missing,
        "failed": failed,
        "dry_run": int(dry_run),
    }


def bootstrap_communes_from_zone_memberships(
    *, dry_run: bool = False
) -> dict[str, int]:
    rows = (
        PlatformZoneMembership.query.filter(
            PlatformZoneMembership.commune_token.like("commune:%")
        )
        .with_entities(PlatformZoneMembership.commune_token)
        .all()
    )
    commune_codes = sorted(
        {
            str(token).split(":", 1)[1].strip()
            for (token,) in rows
            if isinstance(token, str) and token.startswith("commune:")
        }
    )
    created = 0
    updated = 0
    missing = 0
    failed = 0
    for code in commune_codes:
        if not code.isdigit():
            continue
        feature = _fetch_commune_feature(code)
        if not feature:
            missing += 1
            continue
        props = (
            feature.get("properties")
            if isinstance(feature.get("properties"), dict)
            else {}
        )
        label = str((props or {}).get("label") or code).strip()
        name = label.split(",")[0].strip() if label else code
        canton_code = str((props or {}).get("kanton") or "GE").strip().upper() or "GE"
        try:
            canton = GeoUnit.query.filter_by(
                type=GeoUnitType.CANTON, code=canton_code
            ).first()
            if not canton and not dry_run:
                country = GeoUnit.query.filter_by(
                    type=GeoUnitType.COUNTRY, code="CH"
                ).first()
                canton = GeoUnit(
                    type=GeoUnitType.CANTON, code=canton_code, name=canton_code
                )
                if country:
                    canton.parent_id = country.id
                db.session.add(canton)
                db.session.flush()
                created += 1
            commune = GeoUnit.query.filter_by(
                type=GeoUnitType.COMMUNE, code=code
            ).first()
            if not commune and not dry_run:
                commune = GeoUnit(type=GeoUnitType.COMMUNE, code=code, name=name)
                if canton:
                    commune.parent_id = canton.id
                db.session.add(commune)
                created += 1
            elif commune and commune.name != name and not dry_run:
                commune.name = name
                updated += 1
        except Exception:
            failed += 1
            db.session.rollback()
    if not dry_run:
        db.session.commit()
    return {
        "tokens_seen": len(commune_codes),
        "created": created,
        "updated": updated,
        "missing_feature": missing,
        "failed": failed,
        "dry_run": int(dry_run),
    }
