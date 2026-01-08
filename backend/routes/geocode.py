# backend/routes/geocode.py
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple, cast

import requests  # pyright: ignore[reportMissingModuleSource]
from flask import current_app, request  # pyright: ignore[reportMissingImports]
from flask_restx import Namespace, Resource  # pyright: ignore[reportMissingImports]

from services.geolocation.google_places import (
    GooglePlacesError,
    autocomplete_address,
    geocode_address_google,
    get_place_details,
)
from shared.error_handlers import APIErrorHandler

geocode_ns = Namespace(
    "geocode", description="Autocomplete & géocodage avec Google Places API"
)

# Configuration
# Fallback si Google API indisponible
PHOTON = os.getenv("PHOTON_BASE_URL", "https://photon.komoot.io")
USE_GOOGLE_PLACES = os.getenv("USE_GOOGLE_PLACES", "true").lower() in (
    "true",
    "1",
    "yes",
)

# Constantes pour éviter les valeurs magiques
MIN_COORDINATES_COUNT = 2
MIN_QUERY_LENGTH = 2

# Biais géographique Genève (approx)
GENEVA_CENTER: Tuple[float, float] = (46.2044, 6.1432)  # (lat, lon)
GENEVA_BBOX: Tuple[float, float, float, float] = (
    6.02,
    46.16,
    6.27,
    46.28,
)  # (minLon, minLat, maxLon, maxLat)

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


def match_alias(q: str) -> Dict[str, Any] | None:
    q_norm = (q or "").strip()
    for a in ALIASES:
        for pat in a["keys"]:
            if pat.search(q_norm):
                return a
    return None


def looks_like_hospital(q: str) -> bool:
    t = (q or "").lower()
    return any(
        w in t for w in ("hug", "hopital", "hôpital", "hospital", "clinique", "urgenc")
    )


def photon_query(
    q: str, lat: float, lon: float, limit: int, hospital_hint: bool
) -> Dict[str, Any]:
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


def normalize_google_places(
    google_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Normalise les résultats Google Places pour avoir le format "Rue, Numéro, Code Postal, Ville".

    Args:
        google_results: Liste de résultats de autocomplete_address

    Returns:
        Liste de dictionnaires normalisés avec label et address au format complet
    """
    from services.geolocation.google_places import (
        GooglePlacesError,
        extract_address_components,
        get_place_details,
    )

    normalized: List[Dict[str, Any]] = []

    for result in google_results:
        try:
            place_id = result.get("place_id")
            if not place_id:
                continue

            # Récupérer les détails complets du lieu pour obtenir les composants d'adresse
            try:
                details = get_place_details(place_id)
            except GooglePlacesError:
                # Si on ne peut pas récupérer les détails, utiliser les données de base
                description = result.get("description", "")
                main_text = result.get("main_text", "")
                secondary_text = result.get("secondary_text", "")

                # Construire un label basique
                if main_text and secondary_text:
                    label = f"{main_text}, {secondary_text}"
                else:
                    label = description

                normalized.append(
                    {
                        "source": "google",
                        "label": label,
                        "address": description or main_text or label,
                        "lat": None,
                        "lon": None,
                        "place_id": place_id,
                    }
                )
                continue

            # Extraire les composants d'adresse
            address_components = details.get("address_components", [])
            components = extract_address_components(address_components)

            street = components.get("route", "")
            housenumber = components.get("street_number", "")
            city = components.get("locality", "")
            postcode = components.get("postal_code", "")
            place_name = details.get("name", "")

            # Construire l'adresse complète avec numéro et rue
            # Format : "Rue, Numéro" (avec virgule)
            if street and housenumber:
                street_with_number = f"{street}, {housenumber}"
            elif street:
                street_with_number = street
            else:
                street_with_number = None

            # Construire le label : FORCER le format "Rue, Numéro, Code Postal, Ville" (SANS PAYS)
            # ✅ Le code postal doit TOUJOURS être inclus s'il est disponible
            # ❌ Le pays ne doit JAMAIS être inclus dans le label
            if place_name and street_with_number:
                # Lieu nommé avec adresse complète : "Nom, Rue, Numéro, CP, Ville"
                address_parts = [street_with_number]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    address_parts.append(postcode)
                if city:
                    address_parts.append(city)
                # ❌ NE PAS inclure le pays
                address_str = ", ".join(address_parts)
                label = f"{place_name}, {address_str}"
            elif place_name and street:
                # Lieu nommé avec rue mais sans numéro : "Nom, Rue, CP, Ville"
                address_parts = [street]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    address_parts.append(postcode)
                if city:
                    address_parts.append(city)
                # ❌ NE PAS inclure le pays
                address_str = ", ".join(address_parts)
                label = f"{place_name}, {address_str}"
            elif place_name:
                # Lieu nommé sans adresse : juste le nom (fallback)
                label = place_name
            elif street_with_number and city:
                # Adresse complète : "Rue, Numéro, CP, Ville"
                parts = [street_with_number]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    parts.append(postcode)
                if city:
                    parts.append(city)
                # ❌ NE PAS inclure le pays
                label = ", ".join(parts)
            elif street_with_number and postcode:
                # Rue avec numéro et code postal mais sans ville : "Rue, Numéro, CP"
                parts = [street_with_number, postcode]
                label = ", ".join(parts)
            elif street and city:
                # Rue sans numéro : "Rue, CP, Ville"
                parts = [street]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    parts.append(postcode)
                if city:
                    parts.append(city)
                # ❌ NE PAS inclure le pays
                label = ", ".join(parts)
            elif street and postcode:
                # Rue avec code postal mais sans ville : "Rue, CP"
                label = f"{street}, {postcode}"
            elif city:
                # Au moins la ville : inclure le code postal s'il est disponible
                label = f"{postcode} {city}" if postcode and city else city
            elif postcode:
                # Seulement le code postal (cas rare)
                label = postcode
            else:
                # Dernier recours : utiliser l'adresse formatée de Google (sans pays)
                google_address = details.get("address", "")
                # Retirer le pays s'il est présent à la fin
                if google_address:
                    # Retirer "Suisse", "Switzerland", "France", etc. à la fin
                    import re

                    google_address = re.sub(
                        r",?\s*(Suisse|Switzerland|France|Deutschland|Germany|Italy|Italia)\s*$",
                        "",
                        google_address,
                        flags=re.IGNORECASE,
                    ).strip()
                label = google_address or "Adresse"

            # L'adresse à afficher doit toujours inclure le numéro si disponible
            address_display = street_with_number or street or label

            normalized.append(
                {
                    "source": "google",
                    "label": label,
                    "address": address_display,
                    "postcode": postcode,
                    "city": city,
                    "country": components.get("country", ""),
                    "lat": details.get("lat"),
                    "lon": details.get("lon"),
                    "housenumber": housenumber,
                    "place_id": place_id,
                }
            )
        except Exception:
            # Une feature mal formée : on ignore proprement
            continue

    # Priorise les adresses avec n° + CP + label pertinent
    normalized.sort(
        key=lambda r: (
            r.get("housenumber") is None,
            r.get("postcode") is None,
            (r.get("label") or "").lower(),
        )
    )
    return normalized


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
            place_name = props.get("name")

            # ✅ Enrichir avec Google Geocoding si code postal ou numéro manque
            # (seulement si Google Places est activé et qu'on a une rue)
            if (
                USE_GOOGLE_PLACES
                and street
                and city
                and (not postcode or not housenumber)
            ):
                # Construire une adresse de recherche pour Google
                search_address_parts = [street]
                if housenumber:
                    search_address_parts.insert(1, housenumber)
                if city:
                    search_address_parts.append(city)
                if country:
                    search_address_parts.append(country)
                search_address = ", ".join(search_address_parts)

                try:
                    # Appeler Google Geocoding pour enrichir
                    from services.geolocation.google_places import (
                        geocode_address_google,
                    )

                    google_result = geocode_address_google(
                        search_address, country=country or "CH"
                    )
                    if google_result:
                        address_components = google_result.get("address_components", [])
                        # Extraire le code postal si manquant
                        if not postcode:
                            for comp in address_components:
                                if "postal_code" in comp.get("types", []):
                                    postcode = comp.get("long_name")
                                    break
                        # Extraire le numéro si manquant
                        if not housenumber:
                            for comp in address_components:
                                if "street_number" in comp.get("types", []):
                                    housenumber = comp.get("long_name")
                                    break
                except Exception as e:
                    # En cas d'erreur Google, continuer avec les données Photon
                    current_app.logger.debug(
                        "Erreur enrichissement Google pour '%s': %s", search_address, e
                    )

            # Construire l'adresse complète avec numéro et rue
            # Format : "Rue, Numéro" (avec virgule)
            if street and housenumber:
                street_with_number = f"{street}, {housenumber}"
            elif street:
                street_with_number = street
            else:
                street_with_number = None

            # Construire le label : FORCER le format "Rue, Numéro, Code Postal, Ville" (SANS PAYS)
            # ✅ Le code postal doit TOUJOURS être inclus s'il est disponible
            # ❌ Le pays ne doit JAMAIS être inclus dans le label
            # Ne pas inclure les résultats incomplets (sans code postal ET sans numéro si c'est une adresse)
            if place_name and street_with_number:
                # Lieu nommé avec adresse complète : "Nom, Rue, Numéro, CP, Ville"
                address_parts = [street_with_number]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    address_parts.append(postcode)
                if city:
                    address_parts.append(city)
                # ❌ NE PAS inclure le pays
                address_str = ", ".join(address_parts)
                label = f"{place_name}, {address_str}"
            elif place_name and street:
                # Lieu nommé avec rue mais sans numéro : "Nom, Rue, CP, Ville"
                address_parts = [street]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    address_parts.append(postcode)
                if city:
                    address_parts.append(city)
                # ❌ NE PAS inclure le pays
                address_str = ", ".join(address_parts)
                label = f"{place_name}, {address_str}"
            elif place_name:
                # Lieu nommé sans adresse : juste le nom (fallback)
                label = place_name
            elif street_with_number and city:
                # Adresse complète : "Rue, Numéro, CP, Ville"
                # ✅ FORCER le code postal si disponible
                parts = [street_with_number]
                if postcode:
                    parts.append(postcode)
                if city:
                    parts.append(city)
                # ❌ NE PAS inclure le pays
                label = ", ".join(parts)
            elif street_with_number and postcode:
                # Rue avec numéro et code postal mais sans ville : "Rue, Numéro, CP"
                parts = [street_with_number, postcode]
                label = ", ".join(parts)
            elif street and city:
                # Rue sans numéro : "Rue, CP, Ville"
                parts = [street]
                # ✅ Toujours inclure le code postal s'il est disponible
                if postcode:
                    parts.append(postcode)
                if city:
                    parts.append(city)
                # ❌ NE PAS inclure le pays
                label = ", ".join(parts)
            elif street and postcode:
                # Rue avec code postal mais sans ville : "Rue, CP"
                label = f"{street}, {postcode}"
            elif city:
                # Au moins la ville : inclure le code postal s'il est disponible
                label = f"{postcode} {city}" if postcode and city else city
            elif postcode:
                # Seulement le code postal (cas rare)
                label = postcode
            else:
                label = "Adresse"

            # ✅ Ne pas inclure les résultats incomplets pour les adresses
            # (doivent avoir au moins rue + ville, ou lieu nommé)
            if not place_name and not street:
                # Pas de lieu nommé ni de rue : ignorer
                continue

            # L'adresse à afficher doit toujours inclure le numéro si disponible
            address_display = street_with_number or street or label

            out.append(
                {
                    "source": "photon",
                    "label": label,
                    "address": address_display,
                    "postcode": postcode,
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

    # Priorise les adresses avec n° + CP + label pertinent
    out.sort(
        key=lambda r: (
            r.get("housenumber") is None,
            r.get("postcode") is None,
            (r.get("label") or "").lower(),
        )
    )
    return out


@geocode_ns.route("/aliases")
class GeocodeAliases(Resource):
    @geocode_ns.doc(
        security=None,
        params={"q": "Texte à rechercher (ex: HUG, hôpital cantonal, ... )"},
    )
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
                from repositories.favorite_place_repository import (
                    FavoritePlaceRepository,
                )

                favorite_place_repo = FavoritePlaceRepository()
                favs = favorite_place_repo.find_by_company_id_with_label_search(
                    company_id=int(company_id), search_query=q, limit=6
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
                # ✅ FIX: Recherche multi-pays - d'abord Suisse (CH), puis France (FR)
                # Pour la zone frontalière Genève, permettre recherche dans les deux pays
                google_results_ch: List[Dict[str, Any]] = []
                google_results_fr: List[Dict[str, Any]] = []

                # 3a) Recherche en Suisse (CH) en premier
                try:
                    google_results_ch = autocomplete_address(
                        q, country="CH", location={"lat": lat, "lng": lon}, limit=limit
                    )
                    if google_results_ch:
                        current_app.logger.debug(
                            "✅ Google Places (CH) retourne %d résultats pour '%s'",
                            len(google_results_ch),
                            q,
                        )
                except Exception as e_ch:
                    current_app.logger.warning(
                        "⚠️ Erreur Google Places (CH) pour '%s': %s", q, e_ch
                    )

                # 3b) Recherche en France (FR) ensuite (si on n'a pas assez de résultats)
                # On limite à 3 résultats FR pour compléter (max 5 total)
                if len(google_results_ch) < limit:
                    try:
                        fr_limit = max(1, limit - len(google_results_ch))
                        google_results_fr = autocomplete_address(
                            q,
                            country="FR",
                            location={"lat": lat, "lng": lon},
                            limit=fr_limit,
                        )
                        if google_results_fr:
                            current_app.logger.debug(
                                "✅ Google Places (FR) retourne %d résultats pour '%s'",
                                len(google_results_fr),
                                q,
                            )
                    except Exception as e_fr:
                        current_app.logger.warning(
                            "⚠️ Erreur Google Places (FR) pour '%s': %s", q, e_fr
                        )

                # Combiner les résultats : CH en premier, puis FR
                google_results = google_results_ch + google_results_fr

                if google_results:
                    current_app.logger.debug(
                        "✅ Google Places total: %d résultats (%d CH + %d FR) pour '%s'",
                        len(google_results),
                        len(google_results_ch),
                        len(google_results_fr),
                        q,
                    )
                else:
                    current_app.logger.debug(
                        "⚠️ Google Places ne retourne aucun résultat pour '%s'", q
                    )

                for pred in google_results:
                    # Pour chaque prédiction, on peut optionnellement
                    # récupérer les coordonnées via Place Details
                    # (mais c'est plus coûteux en quota)
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

                # ✅ FIX: Si Google Places retourne une liste vide, faire fallback vers Photon
                if not google_results:
                    current_app.logger.debug(
                        "⚠️ Google Places retourne 0 résultats pour '%s', fallback vers Photon",
                        q,
                    )
                    # Fallback vers Photon si Google ne retourne rien
                    try:
                        ph = photon_query(
                            q,
                            lat=lat,
                            lon=lon,
                            limit=limit,
                            hospital_hint=looks_like_hospital(q),
                        )
                        photon_results = normalize_photon(ph)
                        if photon_results:
                            current_app.logger.info(
                                "✅ Photon fallback retourne %d résultats pour '%s'",
                                len(photon_results),
                                q,
                            )
                        results.extend(photon_results)
                    except Exception as e2:
                        current_app.logger.error("❌ Photon autocomplete error: %s", e2)
            except GooglePlacesError as e:
                current_app.logger.warning(
                    "⚠️ Google Places API error, falling back to Photon: %s", e
                )
                # Fallback vers Photon si Google échoue
                try:
                    ph = photon_query(
                        q,
                        lat=lat,
                        lon=lon,
                        limit=limit,
                        hospital_hint=looks_like_hospital(q),
                    )
                    photon_results = normalize_photon(ph)
                    if photon_results:
                        current_app.logger.info(
                            "✅ Photon fallback retourne %d résultats pour '%s'",
                            len(photon_results),
                            q,
                        )
                    results.extend(photon_results)
                except Exception as e2:
                    current_app.logger.error("❌ Photon autocomplete error: %s", e2)
        else:
            # 3) Photon (biais Genève + hint hôpital) - mode fallback
            try:
                ph = photon_query(
                    q,
                    lat=lat,
                    lon=lon,
                    limit=limit,
                    hospital_hint=looks_like_hospital(q),
                )
                photon_results = normalize_photon(ph)
                if photon_results:
                    current_app.logger.debug(
                        "✅ Photon retourne %d résultats pour '%s'",
                        len(photon_results),
                        q,
                    )
                results.extend(photon_results)
            except Exception as e:
                current_app.logger.error("❌ Photon autocomplete error: %s", e)

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
        """Récupère les détails complets d'un lieu
        (coordonnées GPS incluses) via son place_id.
        Utilisé après qu'un utilisateur a sélectionné
        une adresse dans l'autocomplete.
        """
        place_id = request.args.get("place_id", "").strip()

        if not place_id:
            return APIErrorHandler.handle_validation_error(
                "place_id est requis",
                field="place_id",
                logger_instance=current_app.logger,
            )

        if not USE_GOOGLE_PLACES:
            return APIErrorHandler.handle_validation_error(
                "Google Places API non activée",
                logger_instance=current_app.logger,
            )

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
            return APIErrorHandler.handle_exception(e, current_app.logger)


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
            return APIErrorHandler.handle_validation_error(
                "address est requis",
                field="address",
                logger_instance=current_app.logger,
            )

        country = request.args.get("country", "CH")

        try:
            if USE_GOOGLE_PLACES:
                result = geocode_address_google(address, country=country)
            else:
                # Fallback vers le service existant
                from services.geolocation.maps import geocode_address

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
                return APIErrorHandler.handle_not_found(
                    "Coordonnées pour cette adresse",
                    address if "address" in locals() else None,
                    current_app.logger,
                )

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
            return APIErrorHandler.handle_exception(e, current_app.logger)
