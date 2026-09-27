"""Routes API pour OSRM (itinéraires et calculs de durée)."""

import logging
import os

from flask import request  # pyright: ignore[reportMissingImports]
from flask_restx import (  # pyright: ignore[reportMissingImports]
    Namespace,
    Resource,
    fields,
)

from config import Config
from ext import redis_client
from schemas.osrm_schemas import OSRMRouteQuerySchema
from schemas.validation_utils import validate_request
from services.geolocation.historical_eta import get_improved_duration_estimate
from services.geolocation.osrm import route_info
from shared.error_handlers import APIErrorHandler

# OSRM donne le temps à vide. Sans historique de courses, ce facteur rapproche
# l'estimation d'un trajet urbain réel (feux, circulation), sans service externe.
_URBAN_TRAFFIC_FACTOR = 1.55

logger = logging.getLogger(__name__)

osrm_ns = Namespace("osrm", description="OSRM routing services")

# Modèle pour la réponse de route
route_response_model = osrm_ns.model(
    "RouteResponse",
    {
        "duration": fields.Float(description="Durée en secondes"),
        "distance": fields.Float(description="Distance en mètres"),
        "route": fields.List(
            fields.List(fields.Float), description="Liste des coordonnées [lat, lon]"
        ),
    },
)


@osrm_ns.route("/route")
class OSRMRoute(Resource):
    @osrm_ns.doc(
        params={
            "pickup_lat": "Latitude de prise en charge",
            "pickup_lon": "Longitude de prise en charge",
            "dropoff_lat": "Latitude de destination",
            "dropoff_lon": "Longitude de destination",
        }
    )
    @osrm_ns.response(200, "Succès", route_response_model)
    @osrm_ns.response(400, "Paramètres manquants")
    @osrm_ns.response(500, "Erreur serveur")
    def get(self):
        """Obtient l'itinéraire réel entre deux points via OSRM."""
        try:
            # ✅ Validation centralisée avec Marshmallow
            from marshmallow import (  # pyright: ignore[reportMissingImports]
                ValidationError,
            )

            try:
                validated_data = validate_request(
                    OSRMRouteQuerySchema(), dict(request.args)
                )
            except ValidationError as e:
                # handle_exception gère déjà ValidationError avec les messages détaillés
                return APIErrorHandler.handle_exception(e, logger)

            # Extraction des valeurs validées
            pickup_lat = validated_data["pickup_lat"]
            pickup_lon = validated_data["pickup_lon"]
            dropoff_lat = validated_data["dropoff_lat"]
            dropoff_lon = validated_data["dropoff_lon"]

            # URL du serveur OSRM (priorité Config -> env -> fallback service docker)
            config_base = getattr(Config, "UD_OSRM_URL", None)
            osrm_base_url = os.getenv(
                "OSRM_BASE_URL", config_base or "http://osrm:5000"
            )

            # Obtenir l'itinéraire avec géométrie complète
            result = route_info(
                origin=(pickup_lat, pickup_lon),
                destination=(dropoff_lat, dropoff_lon),
                base_url=osrm_base_url,
                profile="driving",
                timeout=8,
                redis_client=redis_client,
                cache_ttl_s=1,
                overview="full",  # Géométrie complète
                geometries="geojson",
                steps=False,
                annotations=False,
            )

            # Extraire les coordonnées de la géométrie
            route_coords = []
            # ✅ P1: Protéger accès dictionnaires pour éviter KeyError
            geometry = result.get("geometry")
            if geometry and isinstance(geometry, dict):
                coordinates = geometry.get("coordinates")
                if coordinates:
                    # OSRM retourne [lon, lat], on convertit en [lat, lon] pour
                    # Leaflet
                    route_coords = [[coord[1], coord[0]] for coord in coordinates]

            is_fallback = bool(result.get("fallback"))
            road_duration = float(result.get("duration") or 0)
            typical_duration = None
            duration_source = "haversine" if is_fallback else "osrm"
            if not is_fallback and road_duration > 0:
                typical_duration, duration_source = get_improved_duration_estimate(
                    (pickup_lat, pickup_lon),
                    (dropoff_lat, dropoff_lon),
                    road_duration,
                    traffic_factor=_URBAN_TRAFFIC_FACTOR,
                    use_weather=False,
                )

            return {
                "duration": road_duration,
                "duration_typical": typical_duration,
                "duration_source": duration_source,
                "fallback": is_fallback,
                "distance": result.get("distance", 0),
                "route": route_coords,
            }, 200

        except Exception as e:
            logger.error("Erreur OSRM route: %s", e)
            return APIErrorHandler.handle_exception(e, logger)
