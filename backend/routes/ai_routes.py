"""Routes /ai/* (estimation d'itinéraire pour le portail client)."""

from __future__ import annotations

import logging

from flask import request
from flask_jwt_extended import jwt_required
from flask_restx import Namespace, Resource, fields

from ext import limiter
from services.external.ai import get_optimized_route
from shared.error_handlers import APIErrorHandler

logger = logging.getLogger(__name__)

ai_ns = Namespace("ai", description="Estimation trajet / aide à la réservation")

_optimized_body = ai_ns.model(
    "OptimizedRouteBody",
    {
        "pickup": fields.String(required=True, description="Adresse ou libellé départ"),
        "dropoff": fields.String(
            required=True, description="Adresse ou libellé arrivée"
        ),
    },
)


@ai_ns.route("/optimized-route")
class OptimizedRouteResource(Resource):
    """POST: géocode pickup/dropoff + OSRM, format attendu par le dashboard client."""

    @jwt_required()
    # Fenêtre courte + plafond horaire : évite les rafales (ex. effets React) tout en
    # restant confortable pour la saisie d'adresses (clé = utilisateur JWT via ext).
    @limiter.limit("45 per minute;800 per hour")
    @ai_ns.expect(_optimized_body, validate=False)
    @ai_ns.response(200, "Itinéraire (polyline + distance/durée)")
    @ai_ns.response(400, "Géocodage ou itinéraire impossible")
    def post(self):
        try:
            payload = request.get_json(silent=True) or {}
            pickup = str(payload.get("pickup") or "").strip()
            dropoff = str(payload.get("dropoff") or "").strip()
            if not pickup or not dropoff:
                return {
                    "message": "Les champs pickup et dropoff sont requis.",
                }, 400

            result = get_optimized_route(pickup, dropoff)
            err = result.get("error")
            if err:
                return {"message": str(err)}, 400

            return result, 200
        except Exception as e:
            logger.exception("optimized-route: %s", e)
            return APIErrorHandler.handle_exception(e, logger)
