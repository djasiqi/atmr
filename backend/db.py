# 1. Importez les dépendances nécessaires
from ext import db, app_logger
from models import Booking # <-- L'IMPORTATION CRUCIALE : on importe le vrai modèle

def une_fonction_qui_cree_une_reservation(data):
    """
    Exemple d'une fonction qui crée et sauvegarde une nouvelle réservation.
    """
    # Note : Assurez-vous d'avoir des ID valides pour client_id et user_id
    client_id_valide = data.get('client_id')
    user_id_valide = data.get('user_id')

    if not all([client_id_valide, user_id_valide]):
        # Gérer l'erreur si les ID sont manquants
        return {"error": "ID client ou utilisateur manquant"}, 400

    try:
        # 2. On utilise le vrai nom du modèle : "Booking"
        nouvelle_reservation = Booking(
            customer_name="John Doe",
            pickup_location="1 Rue de la Paix, 75002 Paris",
            dropoff_location="10 Avenue des Champs-Élysées, 75008 Paris",
            amount=50.0,
            user_id=user_id_valide,
            client_id=client_id_valide
            # ... autres champs requis par votre modèle Booking
        )
        
        db.session.add(nouvelle_reservation)
        db.session.commit()
        app_logger.info(f"Nouvelle réservation {nouvelle_reservation.id} créée avec succès.")
        
        return {"message": "Réservation créée", "booking_id": nouvelle_reservation.id}, 201

    except Exception as e:
        db.session.rollback()
        app_logger.error(f"Erreur lors de la création de la réservation : {e}")
        # Gérer l'erreur (ex: retourner une réponse d'erreur 500)
        return {"error": "Une erreur interne est survenue"}, 500