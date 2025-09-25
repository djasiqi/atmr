from flask import Blueprint, request, jsonify
from flask_restx import Namespace, Resource
from services.ai import get_optimized_route, assign_driver_to_booking, predict_travel_time, get_recommended_routes
from models import Booking
from ext import db

# Création du Blueprint et du Namespace Flask-RESTx
ai_bp = Blueprint("ai", __name__)
ai_ns = Namespace("ai", description="Endpoints pour l'optimisation IA des trajets")

@ai_ns.route("/optimized-route")
class OptimizedRoute(Resource):
    def post(self):
        """
        📌 Endpoint : Optimisation de trajet
        ⚡️ Corps de requête JSON attendu :
        {
            "pickup": "Adresse de départ",
            "dropoff": "Adresse d'arrivée"
        }
        """
        try:
            data = request.get_json()
            pickup = data.get("pickup")
            dropoff = data.get("dropoff")

            if not pickup or not dropoff:
                return {"message": "Les champs 'pickup' et 'dropoff' sont requis."}, 400

            route = get_optimized_route(pickup, dropoff)
            if not route:
                return {"message": "Impossible d'optimiser le trajet."}, 500

            return {"route": route}, 200
        except Exception as e:
            return {"message": f"Erreur : {str(e)}"}, 500

@ai_ns.route("/assign-driver/<int:booking_id>")
class AssignDriver(Resource):
    def post(self, booking_id):
        """
        📌 Endpoint : Assignation IA d'un chauffeur
        ⚡️ Paramètre URL : ID de la réservation
        """
        try:
            assigned_booking = assign_driver_to_booking(booking_id)
            if not assigned_booking:
                return {"message": "Aucun chauffeur disponible."}, 404

            return {"message": f"Chauffeur assigné avec succès pour la réservation {booking_id}."}, 200
        except Exception as e:
            return {"message": f"Erreur : {str(e)}"}, 500

@ai_ns.route("/predict-travel-time")
class PredictTravelTime(Resource):
    def post(self):
        """
        📌 Endpoint : Prédiction du temps de trajet
        ⚡️ Corps de requête JSON attendu :
        {
            "pickup": "Adresse de départ",
            "dropoff": "Adresse d'arrivée"
        }
        """
        try:
            data = request.get_json()
            pickup = data.get("pickup")
            dropoff = data.get("dropoff")

            if not pickup or not dropoff:
                return {"message": "Les champs 'pickup' et 'dropoff' sont requis."}, 400

            estimated_time = predict_travel_time(pickup, dropoff)
            return {"estimated_time": estimated_time}, 200
        except Exception as e:
            return {"message": f"Erreur : {str(e)}"}, 500

@ai_ns.route("/recommended-routes/<int:company_id>")
class RecommendedRoutes(Resource):
    def get(self, company_id):
        """
        📌 Endpoint : Recommandation des trajets les plus fréquents
        ⚡️ Paramètre URL : ID de l'entreprise
        """
        try:
            routes = get_recommended_routes(company_id)
            return {"recommended_routes": routes}, 200
        except Exception as e:
            return {"message": f"Erreur : {str(e)}"}, 500

# Ajout du namespace AI à l’API principale
def register_ai_routes(api):
    api.add_namespace(ai_ns, path="/ai")
