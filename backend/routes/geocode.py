# backend/routes/geocode.py
from __future__ import annotations

from flask import request, current_app
from flask_restx import Namespace, Resource
from typing import Any, Dict, List, Optional, Tuple, cast
from sqlalchemy import func
from models import FavoritePlace
import requests
import os
import re

geocode_ns = Namespace("geocode", description="Autocomplete & géocodage (biais Genève)")

PHOTON = os.getenv("PHOTON_BASE_URL", "https://photon.komoot.io")

# Biais géographique Genève (approx)
GENEVA_CENTER: Tuple[float, float] = (46.2044, 6.1432)   # (lat, lon)
GENEVA_BBOX:   Tuple[float, float, float, float] = (6.02, 46.16, 6.27, 46.28)  # (minLon, minLat, maxLon, maxLat)

# ===== Aliases canoniques (regex précompilées) =====
ALIASES: List[Dict[str, Any]] = [
    {
        "keys": [re.compile(r"\bhug\b", re.I),
                 re.compile(r"h[ôo]pit(?:al|aux).+gen[eè]ve", re.I),
                 re.compile(r"\bh[ôo]pital\s+cantonal\b", re.I)],
        "label": "HUG - Hôpitaux Universitaires de Genève",
        "address": "Rue Gabrielle-Perret-Gentil 4, 1205 Genève",
        "lat": 46.19226,
        "lon": 6.14262,
        "category": "hospital",
    },
    # ➕ Ajoute d'autres alias ici (La Tour, Butini, etc.)
]

def _like_ci(col: Any, like_query: str):
    """LIKE insensible à la casse (équivalent portable à ILIKE)."""
    return func.lower(cast(Any, col)).like(like_query)

def match_alias(q: str) -> Optional[Dict[str, Any]]:
    q_norm = (q or "").strip()
    for a in ALIASES:
        for pat in a["keys"]:
            if pat.search(q_norm):
                return a
    return None

def looks_like_hospital(q: str) -> bool:
    t = (q or "").lower()
    return any(w in t for w in ("hug", "hopital", "hôpital", "hospital", "clinique", "urgenc"))

def photon_query(q: str, lat: float, lon: float, limit: int, hospital_hint: bool) -> Dict[str, Any]:
    params = {
        "q": q,
        "limit": max(1, min(limit, 12)),
        "lang": "fr",
        "lat": lat,
        "lon": lon,
        "bbox": f"{GENEVA_BBOX[0]},{GENEVA_BBOX[1]},{GENEVA_BBOX[2]},{GENEVA_BBOX[3]}",
    }
    if hospital_hint:
        params["osm_tag"] = "amenity:hospital"
    r = requests.get(f"{PHOTON}/api", params=params, timeout=6)
    r.raise_for_status()
    return cast(Dict[str, Any], r.json())

def normalize_photon(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    feats = cast(List[Dict[str, Any]], (data or {}).get("features") or [])
    out: List[Dict[str, Any]] = []
    for f in feats:
        try:
            props = cast(Dict[str, Any], f.get("properties") or {})
            geom = cast(Dict[str, Any], f.get("geometry") or {})
            coords = cast(List[float], geom.get("coordinates") or [])
            if len(coords) < 2:
                continue
            lng, lat = float(coords[0]), float(coords[1])

            housenumber = props.get("housenumber")
            street = props.get("street")
            city = props.get("city") or props.get("locality")
            country = props.get("country")
            label = props.get("name") or "Adresse"
            address_display = " ".join(x for x in [street, housenumber] if x) or street or label

            out.append({
                "source": "photon",
                "label": label,
                "address": address_display,
                "postcode": props.get("postcode"),
                "city": city,
                "country": country,
                "lat": float(lat),
                "lon": float(lng),
                "housenumber": housenumber,
            })
        except Exception:
            # Une feature mal formée : on ignore proprement
            continue

    # Priorise les adresses avec n° + label pertinent
    out.sort(key=lambda r: (r.get("housenumber") is None, (r.get("label") or "").lower()))
    return out

@geocode_ns.route("/aliases")
class GeocodeAliases(Resource):
    @geocode_ns.doc(security=None, params={"q": "Texte à rechercher (ex: HUG, hôpital cantonal, ... )"})
    def get(self):
        q = request.args.get("q", "")
        hit = match_alias(q)
        if not hit:
            return [], 200
        # IMPORTANT : label = address pour écriture directe dans le champ
        return [{
            "source": "alias",
            "label": hit["address"],
            "address": hit["address"],
            "lat": hit["lat"],
            "lon": hit["lon"],
            "category": hit.get("category"),
        }], 200

@geocode_ns.route("/autocomplete")
class GeocodeAutocomplete(Resource):
    @geocode_ns.doc(
        security=None,
        params={
            "q": "Texte à rechercher (≥2 caractères)",
            "lat": "Latitude pour le biais",
            "lon": "Longitude pour le biais",
            "limit": "Nombre max de résultats (def 8, max 12)",
            "company_id": "Optionnel: filtre favoris d'une société",
        },
    )
    def get(self):
        q = (request.args.get("q") or "").strip()
        if len(q) < 2:
            return [], 200

        # Biais (fallback Genève)
        try:
            lat = float(request.args.get("lat", GENEVA_CENTER[0]))
            lon = float(request.args.get("lon", GENEVA_CENTER[1]))
        except Exception:
            lat, lon = GENEVA_CENTER

        # Limite bornée 1..12
        try:
            limit = int(request.args.get("limit", 8))
        except Exception:
            limit = 8
        limit = max(1, min(limit, 12))

        results: List[Dict[str, Any]] = []

        # 1) Alias rapides (HUG…)
        alias = match_alias(q)
        if alias:
            results.append({
                "source": "alias",
                "label": alias["address"],   # label = adresse pour l’UI
                "address": alias["address"],
                "lat": alias["lat"],
                "lon": alias["lon"],
                "category": alias.get("category"),
            })

        # 2) Favoris (optionnel)
        company_id = request.args.get("company_id")
        if company_id:
            try:
                like_q = f"%{q.lower()}%"
                favs = (
                    FavoritePlace.query
                    .filter(
                        cast(Any, FavoritePlace.company_id) == int(company_id),
                        _like_ci(FavoritePlace.label, like_q)
                    )
                    .order_by(cast(Any, FavoritePlace.label).asc())
                    .limit(6)
                    .all()
                )
                for f in favs:
                    results.append({
                        "source": "favorite",
                        "label": f.label,
                        "address": f.address,
                        "lat": f.lat,
                        "lon": f.lon,
                        "category": "favorite",
                    })
            except Exception as e:
                current_app.logger.warning(f"Favorites lookup failed: {e}")

        # 3) Photon (biais Genève + hint hôpital)
        try:
            ph = photon_query(q, lat=lat, lon=lon, limit=limit, hospital_hint=looks_like_hospital(q))
            results.extend(normalize_photon(ph))
        except Exception as e:
            current_app.logger.warning(f"Photon autocomplete error: {e}")

        # 4) Dédup (adresse + coords arrondies)
        seen: set[Tuple[str, float, float]] = set()
        uniq: List[Dict[str, Any]] = []
        for r in results:
            addr_or_label = (r.get("address") or r.get("label") or "").strip()
            lat_v = float(r.get("lat") or 0.0)
            lon_v = float(r.get("lon") or 0.0)
            key = (addr_or_label, round(lat_v, 5), round(lon_v, 5))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)

        return uniq[:limit], 200
