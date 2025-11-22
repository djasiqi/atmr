#!/usr/bin/env python3
"""
Script pour réinitialiser les bookings : supprimer les assignations et réinitialiser driver_id
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path pour importer les modules
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app import create_app  # noqa: E402
from ext import db  # noqa: E402
from models import Assignment, Booking, BookingStatus  # noqa: E402


def reset_bookings():
    """Réinitialise tous les bookings : supprime les assignations et réinitialise driver_id et status"""
    app = create_app()

    with app.app_context():
        try:
            # 1. Compter les bookings avec driver_id
            bookings_with_driver = Booking.query.filter(
                Booking.driver_id.isnot(None)
            ).count()

            # Compter les bookings en PENDING
            bookings_pending = Booking.query.filter(
                Booking.status == BookingStatus.PENDING
            ).count()

            print(f"📊 Bookings avec driver_id: {bookings_with_driver}")
            print(f"📊 Bookings en PENDING: {bookings_pending}")

            # 2. Supprimer toutes les assignations
            deleted_assignments = Assignment.query.delete()
            print(f"🗑️  Supprimé {deleted_assignments} assignations")

            # 3. Réinitialiser driver_id et status des bookings
            # On réinitialise seulement ceux qui ont un driver_id
            bookings_reset = Booking.query.filter(Booking.driver_id.isnot(None)).update(
                {Booking.driver_id: None, Booking.status: BookingStatus.ACCEPTED},
                synchronize_session=False,
            )

            print(
                f"🔄 Réinitialisé {bookings_reset} bookings avec driver_id (driver_id = None, status = ACCEPTED)"
            )

            # 4. Mettre à jour tous les bookings en PENDING vers ACCEPTED
            bookings_pending_to_accepted = Booking.query.filter(
                Booking.status == BookingStatus.PENDING
            ).update(
                {Booking.status: BookingStatus.ACCEPTED}, synchronize_session=False
            )

            print(
                f"🔄 Mis à jour {bookings_pending_to_accepted} bookings de PENDING vers ACCEPTED"
            )

            # 5. Commit
            db.session.commit()
            print("✅ Commit effectué avec succès")

            # 6. Vérification
            remaining_with_driver = Booking.query.filter(
                Booking.driver_id.isnot(None)
            ).count()
            remaining_pending = Booking.query.filter(
                Booking.status == BookingStatus.PENDING
            ).count()
            print("✅ Vérification:")
            print(
                f"   - Bookings avec driver_id restants: {remaining_with_driver} (devrait être 0)"
            )
            print(
                f"   - Bookings en PENDING restants: {remaining_pending} (devrait être 0)"
            )

        except Exception as e:
            db.session.rollback()
            print(f"❌ Erreur: {e}")
            raise


if __name__ == "__main__":
    reset_bookings()
