#!/usr/bin/env python3
"""Script temporaire pour réinitialiser les bookings du 2025-11-04."""

from datetime import datetime
from ext import app, db
from models import Booking, Assignment, BookingStatus

with app.app_context():
    # Récupérer les bookings du 2025-11-04 pour company_id=1
    start_date = datetime(2025, 11, 4, 0, 0, 0)
    end_date = datetime(2025, 11, 5, 0, 0, 0)
    
    bookings = Booking.query.filter(
        Booking.company_id == 1,
        Booking.scheduled_time >= start_date,
        Booking.scheduled_time < end_date
    ).all()
    
    print(f"Trouvé {len(bookings)} bookings à réinitialiser")
    
    # Réinitialiser chaque booking
    booking_ids = []
    for b in bookings:
        booking_ids.append(b.id)
        b.status = BookingStatus.PENDING
        b.driver_id = None
    
    # Supprimer les assignments associés
    deleted_assignments = Assignment.query.filter(
        Assignment.booking_id.in_(booking_ids)
    ).delete()
    
    # Commit
    db.session.commit()
    
    print(f"✅ Réinitialisé {len(bookings)} bookings")
    print(f"✅ Supprimé {deleted_assignments} assignments")
