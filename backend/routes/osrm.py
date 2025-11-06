"""Routes API pour OSRM (itinéraires et calculs de durée)."""
import logging
import os

from flask import request
from flask_restx import Namespace, Resource, fields

from config import Config
from ext import redis_client
from services.osrm_client import route_info

logger = logging.getLogger(__name__)

osrm_ns = Namespace("osrm", description="OSRM routing services")

# Modèle pour la réponse de route
route_response_model = osrm_ns.model(
    "RouteResponse", {
        "duration": fields.Float(
            description="Durée en secondes"), "distance": fields.Float(
                description="Distance en mètres"), "route": fields.List(
                    fields.List(
                        fields.Float), description="Liste des coordonnées [lat, lon]"), })


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
            # Récupérer les paramètres
            pickup_lat = request.args.get("pickup_lat", type=float)
            pickup_lon = request.args.get("pickup_lon", type=float)
            dropoff_lat = request.args.get("dropoff_lat", type=float)
            dropoff_lon = request.args.get("dropoff_lon", type=float)

            if None in (pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
                return {"error": "Paramètres manquants"}, 400

            # URL du serveur OSRM (priorité Config -> env -> fallback service docker)
            config_base = getattr(Config, "UD_OSRM_URL", None)
            osrm_base_url = os.getenv("OSRM_BASE_URL", config_base or "http://osrm:5000")

            # Obtenir l'itinéraire avec géométrie complète
            result = route_info(
                origin=(pickup_lat, pickup_lon),  # type: ignore
                destination=(dropoff_lat, dropoff_lon),  # type: ignore
                base_url=osrm_base_url,
                profile="driving",
                timeout=4,  # ⚡ Réduit à 4s pour fail-fast (cohérent avec frontend)
                redis_client=redis_client,
                cache_ttl_s=1,
                overview="full",  # Géométrie complète
                geometries="geojson",
                steps=False,
                annotations=False,
            )

            # Extraire les coordonnées de la géométrie
            route_coords = []
            if result.get("geometry") and result["geometry"].get(
                    "coordinates"):
                # OSRM retourne [lon, lat], on convertit en [lat, lon] pour
                # Leaflet
                route_coords = [
                    [coord[1], coord[0]]
                    for coord in result["geometry"]["coordinates"]
                ]

            return {
                "duration": result.get("duration", 0),
                "distance": result.get("distance", 0),
                "route": route_coords,
            }, 200

        except Exception as e:

            logger.error("Erreur OSRM route: %s", e)
            return {"error": str(e)}, 500
