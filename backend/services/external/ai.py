# services/ai.py (ou routes/ai.py selon ton projet)
import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv
from geopy.distance import geodesic

from ext import db
from models import Booking, BookingStatus, Company, Driver

load_dotenv()
logger = logging.getLogger(__name__)

# =========================
# Config (aligne avec .env)
# =========================
OSRM_TIMEOUT = int(os.getenv("UD_OSRM_TIMEOUT_SEC") or 15)


def _resolve_osrm_base_url() -> str:
    """Même chaîne d'env que le reste du backend (driver, routes/osrm)."""
    return (
        os.getenv("UD_OSRM_BASE_URL")
        or os.getenv("OSRM_URL")
        or os.getenv("OSRM_BASE_URL")
        or "http://osrm:5000"
    )


# Longueur minimale pour éviter des requêtes vides / bruit (Nominatim / Google)
_MIN_ADDRESS_LEN = 2


# -------------------------------------------------------------------
# Géocodage (Google Maps si clé, sinon Nominatim) — aligné sur le reste de l'API
# -------------------------------------------------------------------
def geocode_address(address: str):
    text = (address or "").strip()
    if len(text) < _MIN_ADDRESS_LEN:
        return None
    try:
        from services.geolocation.maps import geocode_address as maps_geocode

        coords = maps_geocode(text, country="CH")
        if not coords:
            return None
        return (float(coords["lat"]), float(coords["lon"]))
    except ValueError:
        return None
    except Exception as e:
        logger.warning("⚠️ Géocodage (maps): %s", e)
        return None


# -------------------------------------------------------------------
# Routing avec OSRM
# -------------------------------------------------------------------
def osrm_route(lat1, lon1, lat2, lon2):
    """Calcule un itinéraire OSRM entre deux points.

    Args:
        lat1, lon1: Coordonnées du point d'origine
        lat2, lon2: Coordonnées du point de destination

    Returns:
        Dict avec polyline, distance_m, duration_s ou None en cas d'erreur
    """
    # ✅ Validation : Vérifier que tous les paramètres sont valides
    if None in (lat1, lon1, lat2, lon2):
        logger.warning(
            (
                "OSRM route: coordonnées invalides (None détecté). "
                "lat1=%s, lon1=%s, lat2=%s, lon2=%s"
            ),
            lat1,
            lon1,
            lat2,
            lon2,
        )
        return None

    try:
        url = f"{_resolve_osrm_base_url()}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
        params = {
            "overview": "full",
            "geometries": "polyline",
            "alternatives": "false",
            "steps": "false",
        }
        r = requests.get(url, params=params, timeout=OSRM_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        routes = data.get("routes") or []
        if not routes:
            return None
        route = routes[0]
        return {
            "polyline": route.get("geometry"),
            "distance_m": int(route.get("distance", 0)),
            "duration_s": int(route.get("duration", 0)),
        }
    except Exception as e:
        logger.error("Erreur OSRM: %s", e)
        return None


# -------------------------------------------------------------------
# API publiques
# -------------------------------------------------------------------
def is_valid_address(address: str) -> bool:
    return geocode_address(address) is not None


def get_optimized_route(pickup: str, dropoff: str) -> dict[str, Any]:
    a = geocode_address(pickup)
    b = geocode_address(dropoff)
    if not a or not b:
        return {"error": "Géocodage impossible pour l'une des adresses."}

    # route_info: cache Redis, même OSRM que le dispatch, repli Haversine si OSRM down
    from ext import redis_client
    from services.geolocation.osrm import route_info

    info = route_info(
        origin=a,
        destination=b,
        base_url=_resolve_osrm_base_url(),
        profile="driving",
        overview="full",
        geometries="polyline",
        steps=False,
        annotations=False,
        redis_client=redis_client,
        timeout=OSRM_TIMEOUT,
    )
    dist = int(float(info.get("distance") or 0))
    dur = int(float(info.get("duration") or 0))
    geometry = info.get("geometry")

    out: dict[str, Any] = {"distance_m": dist, "duration_s": dur}
    if isinstance(geometry, str) and geometry.strip():
        out["polyline"] = geometry
    elif isinstance(geometry, dict) and geometry.get("coordinates"):
        out["geometry"] = geometry

    if not out.get("polyline") and not out.get("geometry"):
        return {"error": "Pas d'itinéraire disponible."}
    return out


def find_best_driver(pickup_address: str, company_id: int):
    try:
        available = Driver.query.filter_by(company_id=company_id, is_active=True).all()
        if not available:
            return None
        pickup_coords = geocode_address(pickup_address)
        if not pickup_coords:
            return None
        min_km, best = float("inf"), None
        for driver in available:
            # On utilise l'adresse du user lié au driver comme approximation
            if driver.user and getattr(driver.user, "address", None):
                driver_coords = geocode_address(driver.user.address)
                if not driver_coords:
                    continue
                km = geodesic(pickup_coords, driver_coords).km
                if km < min_km:
                    min_km, best = km, driver
        return best
    except Exception as e:
        logger.error("Erreur find_best_driver: %s", e)
        return None


def assign_driver_to_booking(booking_id: int):
    try:
        booking = Booking.query.get(booking_id)
        if not booking:
            return None
        best_driver = find_best_driver(booking.pickup_location, booking.company_id)
        if not best_driver:
            return None
        booking.driver_id = best_driver.id
        booking.status = BookingStatus.ASSIGNED
        db.session.commit()
        return booking
    except Exception as e:
        logger.error("Erreur assign_driver_to_booking: %s", e)
        return None


def predict_travel_time(pickup: str, dropoff: str):
    try:
        a = geocode_address(pickup)
        b = geocode_address(dropoff)
        if not a or not b:
            return {"error": "Adresses invalides."}
        route = osrm_route(a[0], a[1], b[0], b[1])
        if not route:
            return {"error": "Pas de route trouvée."}
        return {"duration_seconds": route["duration_s"]}
    except Exception as e:
        logger.error("Erreur predict_travel_time: %s", e)
        return {"error": "Erreur interne."}


def get_recommended_routes(company_id: int):
    """Retourne le format attendu par le front:
    { "recommended_routes": [ ["A -> B", 12], ... ] }.
    """
    try:
        company = Company.query.get(company_id)
        if not company:
            return {"recommended_routes": []}
        bookings = Booking.query.filter_by(company_id=company_id).all()
        route_counts: dict[str, int] = {}
        for b in bookings:
            key = f"{b.pickup_location} -> {b.dropoff_location}"
            route_counts[key] = route_counts.get(key, 0) + 1
        top = sorted(route_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return {"recommended_routes": top}
    except Exception as e:
        logger.error("Erreur get_recommended_routes: %s", e)
        return {"recommended_routes": []}
