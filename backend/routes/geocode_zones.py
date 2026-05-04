from __future__ import annotations

from typing import Any, Dict

from flask import request
from flask_restx import Resource
from sqlalchemy.orm import selectinload

from models.geo_unit import GeoUnit
from routes.geocode_ns import geocode_ns

from . import geocode as G


@geocode_ns.route("/zones")
class GeocodeZones(Resource):
    @geocode_ns.doc(
        security=None,
        params={
            "q": "Texte recherché (nom de commune/canton/district)",
            "lang": "Langue de recherche (défaut: fr)",
            "types": "Types autorisés (commune,canton,district) - défaut: commune,canton",
            "limit": "Nombre max de résultats (défaut 20, max 50)",
            "canton_code": "Filtre canton (ex: GE) - utile pour charger les communes sur carte",
            "include_geometry": "Inclure la géométrie GeoJSON des communes (plus lourd)",
            "ids": "Hydratation par IDs (ex: 12,45,99)",
            "tokens": "Hydratation par tokens canoniques/legacy",
        },
    )
    def get(self):
        zone_types = G._parse_zone_types(request.args.get("types"))
        zone_ids = G._parse_zone_ids(request.args.get("ids"))
        lang = (request.args.get("lang") or "fr").strip() or "fr"
        sources_used: list[str] = []
        degraded = False
        breaker_open = False
        cache_hit = False

        try:
            limit = int(request.args.get("limit", G.ZONE_DEFAULT_LIMIT))
        except Exception:
            limit = G.ZONE_DEFAULT_LIMIT
        limit = max(1, min(limit, 500))
        canton_code = (request.args.get("canton_code") or "").strip().upper()
        include_geometry = str(
            request.args.get("include_geometry") or ""
        ).strip().lower() in {
            "1",
            "true",
            "yes",
        }

        base_query = GeoUnit.query.options(selectinload(GeoUnit.parent)).filter(
            GeoUnit.type.in_(zone_types)
        )

        if zone_ids:
            records = base_query.filter(GeoUnit.id.in_(zone_ids)).all()
            by_id = {item.id: item for item in records}
            ordered = [by_id[id_] for id_ in zone_ids if id_ in by_id]
            items = [G._serialize_zone_item(item) for item in ordered]
            if items:
                sources_used.append("db")
            return {
                "items": items,
                "meta": {
                    "lang": lang,
                    "degraded": degraded,
                    "breaker_open": breaker_open,
                    "cache_hit": cache_hit,
                    "sources_used": sources_used,
                },
            }, 200

        raw_tokens = request.args.get("tokens")
        if raw_tokens:
            token_values = [v.strip() for v in raw_tokens.split(",") if v.strip()]
            hydrated_items: list[Dict[str, Any]] = []
            for token in token_values:
                canonical = G.ZONE_TOKEN_PATTERN.match(token)
                if canonical:
                    zone_type, code = canonical.group(1), canonical.group(2)
                    item = base_query.filter(
                        GeoUnit.type == G.ZONE_TYPE_MAP[zone_type], GeoUnit.code == code
                    ).first()
                    if item:
                        hydrated_items.append(G._serialize_zone_item(item))
                    else:
                        name = f"{zone_type}:{code}"
                        canton_code = (
                            code
                            if zone_type == "canton" and code in G.SWISS_CANTON_CODES
                            else None
                        )
                        if zone_type == "commune" and code.isdigit():
                            # Fallback to geo.admin commune feature when local GeoUnit is missing.
                            feature = G._fetch_commune_geometry_geojson(code)
                            props = (
                                feature.get("properties")
                                if isinstance(feature, dict)
                                else {}
                            )
                            label = (
                                str((props or {}).get("label") or "").strip()
                                if isinstance(props, dict)
                                else ""
                            )
                            if label:
                                name = label
                            kanton = (
                                str((props or {}).get("kanton") or "").strip().upper()
                                if isinstance(props, dict)
                                else ""
                            )
                            if kanton:
                                canton_code = kanton
                        hydrated_items.append(
                            {
                                "id": int(code)
                                if zone_type == "commune" and code.isdigit()
                                else None,
                                "type": zone_type,
                                "code": code,
                                "name": name,
                                "canton_code": canton_code,
                                "token": token,
                                "source": "geoadmin",
                                "confidence": "inferred",
                            }
                        )
                    continue
                decoded = G._decode_named_zone_token(token)
                if not decoded:
                    continue
                zone_type, zone_name = decoded
                hydrated_items.append(
                    {
                        "id": None,
                        "type": zone_type,
                        "code": None,
                        "name": zone_name,
                        "canton_code": None,
                        "token": token,
                        "source": "photon",
                        "confidence": "fallback",
                    }
                )
            if hydrated_items:
                sources_used.extend(
                    sorted(
                        {
                            item["source"]
                            for item in hydrated_items
                            if item.get("source")
                        }
                    )
                )
                return {
                    "items": hydrated_items,
                    "meta": {
                        "lang": lang,
                        "degraded": degraded,
                        "breaker_open": breaker_open,
                        "cache_hit": cache_hit,
                        "sources_used": sources_used,
                    },
                }, 200

        q = (request.args.get("q") or "").strip()
        if len(q) < G.MIN_QUERY_LENGTH:
            if canton_code:
                candidates = base_query.all()
                filtered = []
                for item in candidates:
                    item_canton = G._resolve_canton_code(item)
                    if item_canton == canton_code:
                        filtered.append(item)
                filtered = sorted(filtered, key=lambda item: (item.name or "").lower())
                items = [G._serialize_zone_item(item) for item in filtered[:limit]]
                if include_geometry:
                    for item in items:
                        if str(item.get("type")) != "commune":
                            continue
                        code = str(item.get("code") or "").strip()
                        geometry = G._fetch_commune_geometry_geojson(code)
                        if geometry:
                            item["geometry"] = geometry.get("geometry")
                if items:
                    sources_used.append("db")
                return {
                    "items": items,
                    "meta": {
                        "lang": lang,
                        "degraded": degraded,
                        "breaker_open": breaker_open,
                        "cache_hit": cache_hit,
                        "sources_used": sources_used,
                        "canton_code": canton_code,
                    },
                }, 200
            return {
                "items": [],
                "meta": {
                    "lang": lang,
                    "degraded": degraded,
                    "breaker_open": breaker_open,
                    "cache_hit": cache_hit,
                    "sources_used": sources_used,
                },
            }, 200

        query_cache_key = (
            f"zones:v={G.ZONE_QUERY_CACHE_VERSION}&q={G._normalize_zone_search_text(q)}"
            f"&lang={lang}&types={','.join(sorted([t.value for t in zone_types]))}"
        )
        cached_items = G._zone_cache_get(query_cache_key)
        if cached_items is not None:
            cache_hit = True
            sources_used.extend(
                sorted(
                    {
                        source
                        for source in [
                            str(item.get("source") or "").strip()
                            for item in cached_items
                        ]
                        if source
                    }
                )
            )
            return {
                "items": cached_items[:limit],
                "meta": {
                    "lang": lang,
                    "degraded": degraded,
                    "breaker_open": breaker_open,
                    "cache_hit": cache_hit,
                    "sources_used": ["cache", *sources_used]
                    if sources_used
                    else ["cache"],
                },
            }, 200

        geoadmin_items, geoadmin_degraded, geoadmin_breaker_open = (
            G._search_geoadmin_zones(q, lang=lang, types=zone_types, limit=limit)
        )
        degraded = degraded or geoadmin_degraded
        breaker_open = breaker_open or geoadmin_breaker_open
        if geoadmin_items:
            sources_used.append("geoadmin")

        q_norm = G._normalize_zone_search_text(q)
        candidates = base_query.all()
        ranked: list[tuple[int, GeoUnit]] = []
        for item in candidates:
            name_norm = G._normalize_zone_search_text(item.name or "")
            if not name_norm:
                continue
            if name_norm == q_norm:
                score = 0
            elif name_norm.startswith(q_norm):
                score = 1
            elif q_norm in name_norm:
                score = 2
            else:
                continue
            ranked.append((score, item))

        ranked.sort(key=lambda x: (x[0], x[1].name.lower()))
        db_items = [G._serialize_zone_item(item) for _, item in ranked[:limit]]
        if db_items:
            sources_used.append("db")

        fallback_items = G._fallback_geocode_zones(q, limit)
        if fallback_items:
            sources_used.append("photon")

        merged_items: list[Dict[str, Any]] = []
        for collection in [geoadmin_items, db_items, fallback_items]:
            merged_items.extend(collection)

        seen_tokens: set[str] = set()
        dedup_items: list[Dict[str, Any]] = []
        for item in merged_items:
            token = str(item.get("token") or "").strip()
            if not token:
                continue
            if token in seen_tokens:
                continue
            seen_tokens.add(token)
            dedup_items.append(item)
            if len(dedup_items) >= limit:
                break

        G._zone_cache_set(query_cache_key, dedup_items, G.ZONE_QUERY_CACHE_TTL_SECONDS)
        return {
            "items": dedup_items,
            "meta": {
                "lang": lang,
                "degraded": degraded,
                "breaker_open": breaker_open,
                "cache_hit": cache_hit,
                "sources_used": sorted(set(sources_used)),
            },
        }, 200
