# backend/routes/geocode.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple, cast

import requests
from flask import current_app, request
from flask_restx import Namespace, Resource
from sqlalchemy import func

from models import FavoritePlace
from services.google_places import GooglePlacesError, autocomplete_address, geocode_address_google, get_place_details

geocode_ns = Namespace("geocode", description="Autocomplete & géocodage avec Google Places API")

# Configuration
# Fallback si Google API indisponible
PHOTON = os.getenv("PHOTON_BASE_URL", "https://photon.komoot.io")
USE_GOOGLE_PLACES = os.getenv("USE_GOOGLE_PLACES", "true").lower() in ("true", "1", "yes")

# Constantes pour éviter les valeurs magiques
MIN_COORDINATES_COUNT = 2
MIN_QUERY_LENGTH = 2

# Biais géographique Genève (approx)
GENEVA_CENTER: Tuple[float, float] = (46.2044, 6.1432)  # (lat, lon)
GENEVA_BBOX: Tuple[float, float, float, float] = (6.02, 46.16, 6.27, 46.28)  # (minLon, minLat, maxLon, maxLat)

# ===== Aliases canoniques (regex précompilées) =====
ALIASES: List[Dict[str, Any]] = [
    {
        "keys": [
            re.compile(r"\bhug\b", re.I),
            re.compile(r"h[ôo]pit(?:al|aux).+gen[eè]ve", re.I),
            re.compile(r"\bh[ôo]pital\s+cantonal\b", re.I),
        ],
        "label": "HUG - Hôpitaux Universitaires de Genève",
        "address": "Rue Gabrielle-Perret-Gentil 4, 1205 Genève",
        "lat": 46.19226,
        "lon": 6.14262,
        "category": "hospital",
    },
    # Ajoute d'autres alias ici (La Tour, Butini, etc.)
]


def _like_ci(col: Any, like_query: str):
    """LIKE insensible à la casse (équivalent portable à ILIKE)."""
    return func.lower(col).like(like_query)


def match_alias(q: str) -> Dict[str, Any] | None:
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
    # Typer correctement params pour satisfaire mypy
    params: dict[str, str | int | float] = {
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
    return cast("Dict[str, Any]", r.json())


def normalize_photon(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    feats = cast("List[Dict[str, Any]]", (data or {}).get("features") or [])
    out: List[Dict[str, Any]] = []
    for f in feats:
        try:
            props = cast("Dict[str, Any]", f.get("properties") or {})
            geom = cast("Dict[str, Any]", f.get("geometry") or {})
            coords = cast("List[float]", geom.get("coordinates") or [])
            if len(coords) < MIN_COORDINATES_COUNT:
                continue
            lng, lat = float(coords[0]), float(coords[1])

            housenumber = props.get("housenumber")
            street = props.get("street")
            city = props.get("city") or props.get("locality")
            postcode = props.get("postcode")
            country = props.get("country")

            # Construire l'adresse complète avec numéro et rue
            street_with_number = " ".join(x for x in [street, housenumber] if x) if street else None

            # Construire le label : nom OU adresse complète OU au moins la
            # ville
            if props.get("name"):
                label = props.get("name")
            elif street_with_number:
                # Adresse complète : "Rue + Numéro, CP, Ville"
                parts = [street_with_number, postcode, city]
                label = ", ".join(x for x in parts if x)
            elif city:
                label = city
            else:
                label = "Adresse"

            address_display = street_with_number or street or label

            out.append(
                {
                    "source": "photon",
                    "label": label,
                    "address": address_display,
                    "postcode": props.get("postcode"),
                    "city": city,
                    "country": country,
                    "lat": float(lat),
                    "lon": float(lng),
                    "housenumber": housenumber,
                }
            )
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
        return [
            {
                "source": "alias",
                "label": hit["address"],
                "address": hit["address"],
                "lat": hit["lat"],
                "lon": hit["lon"],
                "category": hit.get("category"),
            }
        ], 200


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
        if len(q) < MIN_QUERY_LENGTH:
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
            results.append(
                {
                    "source": "alias",
                    "label": alias["address"],  # label = adresse pour l'UI
                    "address": alias["address"],
                    "lat": alias["lat"],
                    "lon": alias["lon"],
                    "category": alias.get("category"),
                }
            )

        # 2) Favoris (optionnel)
        company_id = request.args.get("company_id")
        if company_id:
            try:
                like_q = f"%{q.lower()}%"
                favs = (
                    FavoritePlace.query.filter(
                        cast("Any", FavoritePlace.company_id) == int(company_id), _like_ci(FavoritePlace.label, like_q)
                    )
                    .order_by(cast("Any", FavoritePlace.label).asc())
                    .limit(6)
                    .all()
                )
                for f in favs:
                    results.append(
                        {
                            "source": "favorite",
                            "label": f.label,
                            "address": f.address,
                            "lat": f.lat,
                            "lon": f.lon,
                            "category": "favorite",
                        }
                    )
            except Exception as e:
                current_app.logger.warning("Favorites lookup failed: %s", e)

        # 3) Google Places API (prioritaire) ou fallback Photon
        if USE_GOOGLE_PLACES:
            try:
                # Appel à Google Places Autocomplete
                google_results = autocomplete_address(q, location={"lat": lat, "lng": lon}, limit=limit)

                for pred in google_results:
                    # Pour chaque prédiction, on peut optionnellement récupérer les coordonnées
                    # via Place Details (mais c'est plus coûteux en quota)
                    # Pour l'autocomplete, on retourne juste les suggestions
                    results.append(
                        {
                            "source": "google_places",
                            "label": pred.get("description", ""),
                            "address": pred.get("description", ""),
                            "place_id": pred.get("place_id"),
                            "main_text": pred.get("main_text", ""),
                            "secondary_text": pred.get("secondary_text", ""),
                            "types": pred.get("types", []),
                            # Les coordonnées seront récupérées lors de la
                            # sélection finale
                            "lat": None,
                            "lon": None,
                        }
                    )
            except GooglePlacesError as e:
                current_app.logger.warning("⚠️ Google Places API error, falling back to Photon: %s", e)
                # Fallback vers Photon si Google échoue
                try:
                    ph = photon_query(q, lat=0.0, lon=0.0, limit=limit, hospital_hint=looks_like_hospital(q))
                    results.extend(normalize_photon(ph))
                except Exception as e2:
                    current_app.logger.warning("Photon autocomplete error: %s", e2)
        else:
            # 3) Photon (biais Genève + hint hôpital) - mode fallback
            try:
                ph = photon_query(q, lat=0.0, lon=0.0, limit=limit, hospital_hint=looks_like_hospital(q))
                results.extend(normalize_photon(ph))
            except Exception as e:
                current_app.logger.warning("Photon autocomplete error: %s", e)

        # 4) Dédup (adresse + coords arrondies)
        seen: set[Tuple[str, float, float]] = set()
        uniq: List[Dict[str, Any]] = []
        for r in results:
            addr_or_label = (r.get("address") or r.get("label") or "").strip()
            lat_v = float(r.get("lat") or 0.0) if r.get("lat") is not None else 0.0
            lon_v = float(r.get("lon") or 0.0) if r.get("lon") is not None else 0.0
            # Pour les résultats Google sans coordonnées, utiliser place_id
            # pour dédup
            place_id = r.get("place_id")
            if place_id:
                key = (str(place_id), 0.0, 0.0)
            else:
                key = (addr_or_label or "unknown", round(lat_v, 5), round(lon_v, 5))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(r)

        return uniq[:limit], 200


@geocode_ns.route("/place-details")
class PlaceDetails(Resource):
    @geocode_ns.doc(
        security=None,
        params={
            "place_id": "ID Google Places de l'adresse sélectionnée",
        },
    )
    def get(self):
        """Récupère les détails complets d'un lieu (coordonnées GPS incluses) via son place_id.
        Utilisé après qu'un utilisateur a sélectionné une adresse dans l'autocomplete.
        """
        place_id = request.args.get("place_id", "").strip()

        if not place_id:
            return {"error": "place_id est requis"}, 400

        if not USE_GOOGLE_PLACES:
            return {"error": "Google Places API non activée"}, 503

        try:
            details = get_place_details(place_id)

            return {
                "source": "google_places",
                "place_id": details.get("place_id"),
                "address": details.get("address"),
                "lat": details.get("lat"),
                "lon": details.get("lon"),
                "name": details.get("name"),
                "types": details.get("types", []),
                "address_components": details.get("address_components", []),
            }, 200

        except GooglePlacesError as e:
            current_app.logger.error("❌ Erreur Place Details: %s", e)
            return {"error": str(e)}, 500


@geocode_ns.route("/geocode")
class GeocodeAddress(Resource):
    @geocode_ns.doc(
        security=None,
        params={
            "address": "Adresse complète à géocoder",
            "country": "Code pays (ex: CH) - optionnel",
        },
    )
    def get(self):
        """Géocode une adresse complète et retourne les coordonnées GPS.
        Utilisé lorsqu'une adresse est saisie manuellement (sans autocomplete).
        """
        address = request.args.get("address", "").strip()

        if not address:
            return {"error": "address est requis"}, 400

        country = request.args.get("country", "CH")

        try:
            if USE_GOOGLE_PLACES:
                result = geocode_address_google(address, country=country)
            else:
                # Fallback vers le service existant
                from services.maps import geocode_address

                coords = geocode_address(address, country=country)
                result = (
                    {
                        "address": address,
                        "lat": coords.get("lat"),
                        "lon": coords.get("lon"),
                    }
                    if coords
                    else None
                )

            if not result:
                return {"error": "Aucune coordonnée trouvée pour cette adresse"}, 404

            return {
                "source": "google_geocoding" if USE_GOOGLE_PLACES else "nominatim",
                "address": result.get("address"),
                "lat": result.get("lat"),
                "lon": result.get("lon"),
                "place_id": result.get("place_id"),
                "location_type": result.get("location_type"),
            }, 200

        except Exception as e:
            current_app.logger.error("❌ Erreur géocodage: %s", e)
            return {"error": "Erreur lors du géocodage"}, 500
