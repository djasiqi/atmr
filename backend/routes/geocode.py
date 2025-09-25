# backend/routes/geocode.py
from flask import request, current_app
from flask_restx import Namespace, Resource
from models import db, FavoritePlace  # nécessite que FavoritePlace soit défini dans models.py
import requests, os, re

# RESTX namespace (toutes les routes seront montées sous /api via Api(prefix=...))
geocode_ns = Namespace("geocode", description="Autocomplete & géocodage (biais Genève)")

PHOTON = os.getenv("PHOTON_BASE_URL", "https://photon.komoot.io")

# Biais géographique Genève (approx)
GENEVA_CENTER = (46.2044, 6.1432)               # (lat, lon)
GENEVA_BBOX   = (6.02, 46.16, 6.27, 46.28)      # (minLon, minLat, maxLon, maxLat)

# ===== Aliases canoniques =====
ALIASES = [
    {
        # HUG – variantes usuelles (accents tolérés)
        "keys": [r"\bhug\b", r"h[ôo]pit(?:al|aux).+gen[eè]ve", r"h[ôo]pital\s+cantonal"],
        "label": "HUG - Hôpitaux Universitaires de Genève",
        "address": "Rue Gabrielle-Perret-Gentil 4, 1205 Genève",
        "lat": 46.19226,
        "lon": 6.14262,
        "category": "hospital",
    },
    # ➕ Ajoute ici d'autres alias (Hôpital de la Tour, Butini, etc.)
]

def match_alias(q: str):
    q_norm = (q or "").lower().strip()
    for a in ALIASES:
        for k in a["keys"]:
            if re.search(k, q_norm):
                return a
    return None

def looks_like_hospital(q: str) -> bool:
    t = (q or "").lower()
    return any(w in t for w in ("hug", "hopital", "hôpital", "hospital", "clinique", "urgenc"))

def photon_query(q: str, lat: float, lon: float, limit: int, hospital_hint: bool):
    params = {
        "q": q,
        "limit": max(1, min(limit, 12)),
        "lang": "fr",
        "lat": lat,
        "lon": lon,
        "bbox": f"{GENEVA_BBOX[0]},{GENEVA_BBOX[1]},{GENEVA_BBOX[2]},{GENEVA_BBOX[3]}",
    }
    # Filtre hospital si le texte le suggère (améliore la précision)
    if hospital_hint:
        params["osm_tag"] = "amenity:hospital"
    r = requests.get(f"{PHOTON}/api", params=params, timeout=6)
    r.raise_for_status()
    return r.json()

def normalize_photon(data):
    feats = (data or {}).get("features") or []
    out = []
    for f in feats:
        props = f.get("properties", {}) or {}
        lng, lat = f["geometry"]["coordinates"]
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
            return []
        # IMPORTANT : label = address pour écriture directe dans le champ
        return [{
            "source": "alias",
            "label": hit["address"],
            "address": hit["address"],
            "lat": hit["lat"],
            "lon": hit["lon"],
            "category": hit.get("category"),
        }]

@geocode_ns.route("/autocomplete")
class GeocodeAutocomplete(Resource):
    @geocode_ns.doc(
        security=None,
        params={
            "q": "Texte à rechercher (≥2 caractères)",
            "lat": "Latitude pour le biais",
            "lon": "Longitude pour le biais",
            "limit": "Nombre max de résultats (def 8)",
            "company_id": "Optionnel: filtre favoris d'une société",
        },
    )
    def get(self):
        q = (request.args.get("q") or "").strip()
        if len(q) < 2:
            return []

        # Biais (fallback Genève)
        try:
            lat = float(request.args.get("lat", GENEVA_CENTER[0]))
            lon = float(request.args.get("lon", GENEVA_CENTER[1]))
        except ValueError:
            lat, lon = GENEVA_CENTER

        try:
            limit = int(request.args.get("limit", 8))
        except ValueError:
            limit = 8

        results = []

        # 1) Alias HUG (robuste)
        q_norm = q.lower()
        if any(re.search(pat, q_norm) for pat in [
            r"\bhug\b", r"h[ôo]pit(?:al|aux).+gen[eè]ve", r"\bh[ôo]pital\s+cantonal\b"
        ]):
            results.append({
                "source": "alias",
                "label": "Rue Gabrielle-Perret-Gentil 4, 1205 Genève",  # on force le label = adresse
                "address": "Rue Gabrielle-Perret-Gentil 4, 1205 Genève",
                "lat": 46.19226,
                "lon": 6.14262,
                "category": "hospital",
            })

        # 2) Favoris (optionnel)
        company_id = request.args.get("company_id")
        if company_id:
            try:
                favs = (
                    FavoritePlace.query
                    .filter(FavoritePlace.company_id == int(company_id),
                            FavoritePlace.label.ilike(f"%{q}%"))
                    .order_by(FavoritePlace.label.asc())
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

        # 4) Dédup (adresse/coords)
        seen, uniq = set(), []
        for r in results:
            key = ((r.get("address") or r.get("label")), round(r.get("lat", 0), 5), round(r.get("lon", 0), 5))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)

        return uniq[:max(limit, 8)]
