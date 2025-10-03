# 1. Importez les dépendances nécessaires
from ext import db, app_logger
from models import Booking  
from typing import Any, Dict, cast


def une_fonction_qui_cree_une_reservation(data: Dict[str, Any]):
    """
    Exemple d'une fonction qui crée et sauvegarde une nouvelle réservation.
    """
    # Note : Assurez-vous d'avoir des ID valides pour client_id et user_id
    client_id_raw = data.get("client_id")
    user_id_raw = data.get("user_id")

    # ✅ Évite None explicitement (sinon Pylance considère Any|None -> int invalide)
    if client_id_raw is None or user_id_raw is None:
        return {"error": "ID client ou utilisateur manquant"}, 400

    # Validation/cast robuste (int accepte str → typage propre)
    try:
        client_id = int(str(client_id_raw))
        user_id = int(str(user_id_raw))
    except (ValueError, TypeError):
        return {"error": "ID client ou utilisateur invalide"}, 400

    try:
        # Payload typé (Pylance-friendly) puis cast au moment de l'appel
        payload: Dict[str, Any] = {
            "customer_name": str(data.get("customer_name") or "John Doe"),
            "pickup_location": str(data.get("pickup_location") or "1 Rue de la Paix, 75002 Paris"),
            "dropoff_location": str(data.get("dropoff_location") or "10 Avenue des Champs-Élysées, 75008 Paris"),
            "amount": float(data.get("amount") or 50.0),
            "user_id": user_id,
            "client_id": client_id,
        }

        # Optionnels si présents
        company_id_raw = data.get("company_id")
        if company_id_raw is not None:
            try:
                payload["company_id"] = int(str(company_id_raw))
            except (ValueError, TypeError):
                pass
        if "scheduled_time" in data and data["scheduled_time"]:
            # Laisse le validator du modèle gérer le parse/validation
            payload["scheduled_time"] = data["scheduled_time"]
        if "medical_facility" in data:
            payload["medical_facility"] = data["medical_facility"]
        if "doctor_name" in data:
            payload["doctor_name"] = data["doctor_name"]
        if "notes_medical" in data:
            payload["notes_medical"] = data["notes_medical"]

        # Pylance n'a pas de signature kwargs pour SQLAlchemy -> cast/ignore
        BookingCtor = cast(Any, Booking)
        nouvelle_reservation = BookingCtor(**payload)
       
        db.session.add(nouvelle_reservation)
        db.session.commit()
        app_logger.info(f"Nouvelle réservation {nouvelle_reservation.id} créée avec succès.")
        
        return {"message": "Réservation créée", "booking_id": nouvelle_reservation.id}, 201

    except Exception as e:
        db.session.rollback()
        app_logger.error(f"Erreur lors de la création de la réservation : {e}")
        # Gérer l'erreur (ex: retourner une réponse d'erreur 500)
        return {"error": "Une erreur interne est survenue"}, 500