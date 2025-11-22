# backend/services/google_places.py
"""Service pour l'intégration avec Google Places API et Geocoding API.
Gère l'autocomplete d'adresses et le géocodage avec validation des résultats.
"""

import logging
import os
from typing import Any, Dict, List

import requests

app_logger = logging.getLogger("app")

# Clé API Google (Geocoding + Places)
GOOGLE_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

# Configuration par défaut pour la Suisse/Genève
DEFAULT_COUNTRY = "CH"
DEFAULT_LANGUAGE = "fr"
GENEVA_LOCATION = {"lat": 46.2044, "lng": 6.1432}  # Centre de Genève
GENEVA_RADIUS = 50000  # 50km autour de Genève

# Timeout pour les requêtes API
API_TIMEOUT = 8
MIN_QUERY_LENGTH = 2


class GooglePlacesError(Exception):
    """Exception levée lors d'erreurs avec l'API Google Places."""


def autocomplete_address(
    query: str,
    *,
    country: str = DEFAULT_COUNTRY,
    language: str = DEFAULT_LANGUAGE,
    location: Dict[str, float] | None = None,
    radius: int = GENEVA_RADIUS,
    types: str = "(regions)",  # noqa: ARG001
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Autocomplete d'adresses avec Google Places API (Autocomplete).

    Args:
        query: Texte de recherche (minimum 2 caractères)
        country: Code pays ISO (ex: "CH" pour Suisse)
        language: Langue des résultats (ex: "fr")
        location: Centre de recherche {"lat": float, "lng": float}
        radius: Rayon de recherche en mètres
        types: Type de résultats - "(regions)" pour tout type (adresses + établissements)
        limit: Nombre maximum de résultats

    Returns:
        Liste de dictionnaires contenant:
        - place_id: ID unique Google Places
        - description: Adresse formatée complète
        - main_text: Texte principal (rue + numéro)
        - secondary_text: Texte secondaire (ville, pays)
        - types: Liste des types de lieu

    Raises:
        GooglePlacesError: Si la requête échoue

    """
    if not GOOGLE_API_KEY:
        msg = "Clé API Google Maps non configurée (GOOGLE_MAPS_API_KEY)"
        raise GooglePlacesError(msg)

    query = (query or "").strip()
    if len(query) < MIN_QUERY_LENGTH:
        return []

    # URL de l'API Autocomplete
    url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"

    # Paramètres de la requête
    params: Dict[str, Any] = {
        "input": query,
        "key": GOOGLE_API_KEY,
        "language": language,
        "components": f"country:{country}",
    }

    # Ne pas ajouter "types" pour permettre la recherche dans tous les types
    # (adresses, établissements, POI, etc.)

    # Biais géographique (priorité aux résultats proches)
    if location is None:
        location = GENEVA_LOCATION

    params["location"] = f"{location['lat']},{location['lng']}"
    params["radius"] = radius

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get("status") not in ("OK", "ZERO_RESULTS"):
            error_msg = data.get("error_message", data.get("status", "Unknown error"))
            app_logger.warning("⚠️ Google Places Autocomplete: %s", error_msg)
            return []

        predictions = data.get("predictions", [])

        # Formater les résultats
        results = []
        for pred in predictions[:limit]:
            results.append(
                {
                    "place_id": pred.get("place_id"),
                    "description": pred.get("description"),
                    "main_text": pred.get("structured_formatting", {}).get(
                        "main_text", ""
                    ),
                    "secondary_text": pred.get("structured_formatting", {}).get(
                        "secondary_text", ""
                    ),
                    "types": pred.get("types", []),
                }
            )

        return results

    except requests.RequestException as e:
        app_logger.error("❌ Erreur Google Places Autocomplete: %s", e)
        msg = f"Erreur lors de l'autocomplete: {e}"
        raise GooglePlacesError(msg) from e


def get_place_details(
    place_id: str, *, language: str = DEFAULT_LANGUAGE, fields: List[str] | None = None
) -> Dict[str, Any]:
    """Récupère les détails complets d'un lieu via son place_id.

    Args:
        place_id: ID unique Google Places
        language: Langue des résultats
        fields: Liste des champs à récupérer (None = tous les champs basiques)

    Returns:
        Dictionnaire contenant:
        - address: Adresse formatée complète
        - lat: Latitude
        - lon: Longitude
        - address_components: Composants détaillés de l'adresse
        - place_id: ID du lieu
        - types: Types du lieu

    Raises:
        GooglePlacesError: Si la requête échoue

    """
    if not GOOGLE_API_KEY:
        msg = "Clé API Google Maps non configurée"
        raise GooglePlacesError(msg)

    if not place_id:
        msg = "place_id est requis"
        raise ValueError(msg)

    url = "https://maps.googleapis.com/maps/api/place/details/json"

    # Champs par défaut optimisés pour les adresses
    if fields is None:
        fields = [
            "formatted_address",
            "geometry",
            "address_components",
            "place_id",
            "types",
            "name",
        ]

    params = {
        "place_id": place_id,
        "key": GOOGLE_API_KEY,
        "language": language,
        "fields": ",".join(fields),
    }

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "OK":
            error_msg = data.get("error_message", data.get("status"))
            msg = f"Erreur Place Details: {error_msg}"
            raise GooglePlacesError(msg)

        result = data.get("result", {})
        location = result.get("geometry", {}).get("location", {})

        return {
            "address": result.get("formatted_address", ""),
            "lat": location.get("lat"),
            "lon": location.get("lng"),
            "address_components": result.get("address_components", []),
            "place_id": result.get("place_id"),
            "types": result.get("types", []),
            "name": result.get("name", ""),
        }

    except requests.RequestException as e:
        app_logger.error("❌ Erreur Google Place Details: %s", e)
        msg = f"Erreur lors de la récupération des détails: {e}"
        raise GooglePlacesError(msg) from e


def geocode_address_google(
    address: str,
    *,
    country: str | None = DEFAULT_COUNTRY,
    language: str = DEFAULT_LANGUAGE,
) -> Dict[str, Any] | None:
    """Géocode une adresse avec Google Geocoding API.

    Args:
        address: Adresse à géocoder
        country: Code pays pour filtrer les résultats
        language: Langue des résultats

    Returns:
        Dictionnaire contenant:
        - address: Adresse formatée
        - lat: Latitude
        - lon: Longitude
        - place_id: ID Google Places
        - location_type: Type de précision (ROOFTOP, RANGE_INTERPOLATED, etc.)
        - address_components: Composants de l'adresse

        None si aucun résultat trouvé

    Raises:
        GooglePlacesError: Si la requête échoue

    """
    if not GOOGLE_API_KEY:
        msg = "Clé API Google Maps non configurée"
        raise GooglePlacesError(msg)

    address = (address or "").strip()
    if not address:
        msg = "Adresse vide ou invalide"
        raise ValueError(msg)

    url = "https://maps.googleapis.com/maps/api/geocode/json"

    params: Dict[str, Any] = {
        "address": address,
        "key": GOOGLE_API_KEY,
        "language": language,
    }

    if country:
        params["components"] = f"country:{country}"

    try:
        response = requests.get(url, params=params, timeout=API_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get("status") == "ZERO_RESULTS":
            app_logger.warning("⚠️ Aucun résultat de géocodage pour: %s", address)
            return None

        if data.get("status") != "OK":
            error_msg = data.get("error_message", data.get("status"))
            app_logger.warning("⚠️ Google Geocoding: %s", error_msg)
            return None

        results = data.get("results", [])
        if not results:
            return None

        # Prendre le premier résultat (le plus pertinent)
        result = results[0]
        location = result.get("geometry", {}).get("location", {})

        return {
            "address": result.get("formatted_address", ""),
            "lat": location.get("lat"),
            "lon": location.get("lng"),
            "place_id": result.get("place_id"),
            "location_type": result.get("geometry", {}).get("location_type"),
            "address_components": result.get("address_components", []),
        }

    except requests.RequestException as e:
        app_logger.error("❌ Erreur Google Geocoding API: %s", e)
        msg = f"Erreur lors du géocodage: {e}"
        raise GooglePlacesError(msg) from e


def extract_address_components(
    address_components: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Extrait les composants utiles d'une adresse Google.

    Args:
        address_components: Liste des composants d'adresse retournés par Google

    Returns:
        Dictionnaire avec: street_number, route, locality, postal_code, country

    """
    components = {
        "street_number": "",
        "route": "",
        "locality": "",
        "postal_code": "",
        "country": "",
        "administrative_area_level_1": "",  # Canton/État
    }

    for component in address_components:
        types = component.get("types", [])
        value = component.get("long_name", "")

        if "street_number" in types:
            components["street_number"] = value
        elif "route" in types:
            components["route"] = value
        elif "locality" in types:
            components["locality"] = value
        elif "postal_code" in types:
            components["postal_code"] = value
        elif "country" in types:
            components["country"] = value
        elif "administrative_area_level_1" in types:
            components["administrative_area_level_1"] = value

    return components


def geocode_and_validate(
    address: str, *, country: str = DEFAULT_COUNTRY, require_street_number: bool = False
) -> Dict[str, Any] | None:
    """Géocode une adresse et valide qu'elle est suffisamment précise.

    Args:
        address: Adresse à géocoder
        country: Code pays
        require_street_number: Si True, rejette les adresses sans numéro de rue

    Returns:
        Résultat du géocodage avec validation, ou None si invalide

    """
    result = geocode_address_google(address, country=country)

    if not result:
        return None

    # Vérifier la précision de la géolocalisation
    location_type = result.get("location_type", "")
    if location_type not in ("ROOFTOP", "RANGE_INTERPOLATED", "GEOMETRIC_CENTER"):
        app_logger.warning(
            "⚠️ Géolocalisation imprécise (%s) pour: %s", location_type, address
        )

    # Extraire les composants
    components = extract_address_components(result.get("address_components", []))

    # Valider la présence d'un numéro de rue si requis
    if require_street_number and not components.get("street_number"):
        app_logger.warning("⚠️ Adresse sans numéro de rue: %s", address)
        return None

    result["components"] = components
    return result


# Alias pour compatibilité avec l'ancien code
geocode_address = geocode_address_google
