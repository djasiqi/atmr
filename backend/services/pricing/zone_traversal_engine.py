from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from ext import db

ZONE_TRAVERSAL_CORRIDOR_METERS = float(os.getenv("PRICING_ZONE_TRAVERSAL_CORRIDOR_M", "25"))
ZONE_TRAVERSAL_SAMPLE_STEP_METERS = float(os.getenv("PRICING_ZONE_TRAVERSAL_SAMPLE_STEP_M", "200"))
ZONE_TRAVERSAL_SIMPLIFY_TOLERANCE = float(os.getenv("PRICING_ZONE_TRAVERSAL_SIMPLIFY_DEG", "0.0001"))
MIN_LINESTRING_COORDS = 2


@dataclass
class ZoneTraversalResult:
    zones_count: int | None
    zone_ids: list[str]
    confidence: str
    blocking_reasons: list[str]
    source: str


def _to_geojson_linestring(route_geometry: dict[str, Any] | None) -> str | None:
    if not isinstance(route_geometry, dict):
        return None
    gtype = str(route_geometry.get("type") or "")
    coords = route_geometry.get("coordinates")
    if gtype != "LineString" or not isinstance(coords, list) or len(coords) < MIN_LINESTRING_COORDS:
        return None
    return json.dumps({"type": "LineString", "coordinates": coords}, separators=(",", ":"))


def _has_geo_unit_geom_column() -> bool:
    # Feature detection: allows safe runtime when PostGIS columns are absent.
    query = text(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'geo_unit'
          AND column_name IN ('geom', 'geometry')
        LIMIT 1
        """
    )
    try:
        value = db.session.execute(query).scalar()
        return bool(value)
    except Exception:
        return False


def compute_zone_traversal(
    *,
    zone_set_key: str | None,
    route_geometry: dict[str, Any] | None,
    corridor_meters: float = ZONE_TRAVERSAL_CORRIDOR_METERS,
) -> ZoneTraversalResult:
    key_value = str(zone_set_key or "").strip()
    if not key_value:
        return ZoneTraversalResult(
            zones_count=None,
            zone_ids=[],
            confidence="blocked",
            blocking_reasons=["zone_set_missing"],
            source="zone_traversal",
        )

    route_geojson = _to_geojson_linestring(route_geometry)
    if not route_geojson:
        return ZoneTraversalResult(
            zones_count=None,
            zone_ids=[],
            confidence="blocked",
            blocking_reasons=["route_geometry_missing"],
            source="zone_traversal",
        )

    if not _has_geo_unit_geom_column():
        return ZoneTraversalResult(
            zones_count=None,
            zone_ids=[],
            confidence="blocked",
            blocking_reasons=["zone_geometry_unavailable"],
            source="zone_traversal",
        )

    sql = text(
        """
        WITH input_route AS (
            SELECT ST_SetSRID(ST_GeomFromGeoJSON(:route_geojson), 4326) AS geom
        ),
        simplified_route AS (
            SELECT ST_SimplifyPreserveTopology(geom, :simplify_tolerance) AS geom
            FROM input_route
        ),
        route_corridor AS (
            SELECT ST_Buffer(geom::geography, :corridor_meters)::geometry AS geom
            FROM simplified_route
        ),
        intersected_zones AS (
            SELECT
                pzm.zone_id::text AS zone_id
            FROM platform_zone_set pzs
            JOIN platform_zone_membership pzm ON pzm.zone_set_id = pzs.id
            JOIN geo_unit gu
              ON gu.code = split_part(pzm.commune_token, ':', 2)
             AND gu.type = 'commune'
            WHERE pzs.key = :zone_set_key
              AND pzs.is_active = TRUE
              AND pzm.commune_token LIKE 'commune:%'
              AND gu.geom && ST_Envelope((SELECT geom FROM route_corridor))
              AND ST_Intersects(gu.geom, (SELECT geom FROM route_corridor))
        ),
        ranked AS (
            SELECT
                iz.zone_id,
                MIN(
                    ST_LineLocatePoint(
                        (SELECT geom FROM simplified_route),
                        ST_ClosestPoint((SELECT geom FROM simplified_route), gu.geom)
                    )
                ) AS first_pos
            FROM intersected_zones iz
            JOIN platform_zone_membership pzm
              ON pzm.zone_id::text = iz.zone_id
            JOIN platform_zone_set pzs
              ON pzs.id = pzm.zone_set_id
             AND pzs.key = :zone_set_key
             AND pzs.is_active = TRUE
            JOIN geo_unit gu
              ON gu.code = split_part(pzm.commune_token, ':', 2)
             AND gu.type = 'commune'
             AND pzm.commune_token LIKE 'commune:%'
             AND ST_Intersects(gu.geom, (SELECT geom FROM simplified_route))
            GROUP BY iz.zone_id
        ),
        fallback_ranked AS (
            SELECT
                iz.zone_id,
                2.0 AS first_pos
            FROM intersected_zones iz
            WHERE NOT EXISTS (SELECT 1 FROM ranked r WHERE r.zone_id = iz.zone_id)
        )
        SELECT zone_id
        FROM (
            SELECT zone_id, first_pos FROM ranked
            UNION ALL
            SELECT zone_id, first_pos FROM fallback_ranked
        ) ordered
        ORDER BY first_pos ASC, zone_id ASC
        """
    )

    try:
        rows = db.session.execute(
            sql,
            {
                "route_geojson": route_geojson,
                "corridor_meters": max(float(corridor_meters or 0), 1.0),
                "sample_step_m": max(float(ZONE_TRAVERSAL_SAMPLE_STEP_METERS), 25.0),
                "simplify_tolerance": max(float(ZONE_TRAVERSAL_SIMPLIFY_TOLERANCE), 0.00001),
                "zone_set_key": key_value,
            },
        ).fetchall()
    except Exception:
        db.session.rollback()
        return ZoneTraversalResult(
            zones_count=None,
            zone_ids=[],
            confidence="blocked",
            blocking_reasons=["zone_traversal_query_failed"],
            source="zone_traversal",
        )

    zone_ids: list[str] = []
    for row in rows:
        value = str(getattr(row, "zone_id", row[0]) or "").strip()
        if value and value not in zone_ids:
            zone_ids.append(value)

    if not zone_ids:
        return ZoneTraversalResult(
            zones_count=None,
            zone_ids=[],
            confidence="blocked",
            blocking_reasons=["zone_traversal_empty"],
            source="zone_traversal",
        )

    return ZoneTraversalResult(
        zones_count=max(len(zone_ids), 1),
        zone_ids=zone_ids,
        confidence="exact",
        blocking_reasons=[],
        source="postgis_corridor",
    )

