#!/usr/bin/env python
"""
Script pour annuler toutes les assignations d'une journée
Usage: python scripts/reset_assignments.py [YYYY-MM-DD]
"""
import os
import sys

# Ajouter le répertoire backend au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime

from app import create_app
from ext import db
from models import Booking, BookingStatus
from models.dispatch import Assignment
from shared.time_utils import day_local_bounds


def reset_assignments(date_str=None):
    """Annule toutes les assignations pour une date donnée."""
    app = create_app()

    with app.app_context():
        # Date par défaut = aujourd'hui
        if not date_str:
            date_str = datetime.now().strftime('%Y-%m-%d')

        print(f"🔄 Annulation des assignations pour le {date_str}...")

        # Récupérer toutes les réservations de la journée avec statut ASSIGNED
        try:
            start_local, end_local = day_local_bounds(date_str)

            # Récupérer TOUTES les réservations avec un driver_id (ASSIGNED ou ACCEPTED)
            bookings = Booking.query.filter(
                Booking.scheduled_time >= start_local,
                Booking.scheduled_time < end_local,
                Booking.driver_id.isnot(None),
                Booking.status.in_([BookingStatus.ASSIGNED, BookingStatus.ACCEPTED])
            ).all()

            print(f"📋 Trouvé {len(bookings)} réservation(s) avec chauffeur assigné")

            if len(bookings) == 0:
                print("✅ Aucune assignation à annuler")
                return

            # Supprimer les assignations de la table dispatch_assignment
            assignment_ids = []
            for booking in bookings:
                # Trouver l'assignation correspondante
                assignment = Assignment.query.filter_by(booking_id=booking.id).first()
                if assignment:
                    assignment_ids.append(assignment.id)

            if assignment_ids:
                deleted = Assignment.query.filter(Assignment.id.in_(assignment_ids)).delete(synchronize_session=False)
                print(f"🗑️  Supprimé {deleted} assignation(s) de dispatch_assignment")

            # Annuler les assignations sur les bookings
            count = 0
            for booking in bookings:
                driver_name = booking.driver.user.username if booking.driver else 'N/A'
                print(f"  - Réservation #{booking.id} ({booking.customer_full_name}): {driver_name} → Non assigné")
                booking.driver_id = None
                booking.status = BookingStatus.ACCEPTED
                count += 1

            # Sauvegarder
            db.session.commit()
            print(f"\n✅ {count} assignation(s) booking annulée(s) avec succès !")

        except Exception as e:
            db.session.rollback()
            print(f"\n❌ Erreur: {e}")
            raise

if __name__ == '__main__':
    date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    reset_assignments(date_arg)

