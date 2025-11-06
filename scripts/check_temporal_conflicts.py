#!/usr/bin/env python3
"""Script pour vérifier les conflits temporels dans les assignations.

Analyse les assignations d'une company pour détecter les conflits temporels
entre courses assignées au même chauffeur.
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ajouter le répertoire parent au path pour les imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask
from ext import db
from models import Company, Booking, Assignment, Driver
from models.enums import AssignmentStatus
from shared.geo_utils import haversine_minutes
from services.unified_dispatch import settings as ud_settings


def check_temporal_conflicts(company_id: int):
    """Vérifie les conflits temporels pour une company."""
    app = Flask(__name__)
    app.config.from_object("config.Config")
    db.init_app(app)
    
    with app.app_context():
        company = Company.query.get(company_id)
        if not company:
            print(f"❌ Company {company_id} non trouvée")
            return
        
        # Récupérer les paramètres configurables
        dispatch_settings = ud_settings.for_company(company)
        pickup_service_min = dispatch_settings.service_times.pickup_service_min
        dropoff_service_min = dispatch_settings.service_times.dropoff_service_min
        min_transition_margin_min = dispatch_settings.service_times.min_transition_margin_min
        
        print(f"📋 Vérification des conflits temporels pour Company #{company_id}")
        print(f"   Paramètres: pickup={pickup_service_min}min, dropoff={dropoff_service_min}min, marge={min_transition_margin_min}min\n")
        
        # Récupérer toutes les assignations actives
        assignments = (
            Assignment.query.join(Booking)
            .filter(
                Assignment.company_id == company_id,
                Assignment.status.in_([
                    AssignmentStatus.SCHEDULED,
                    AssignmentStatus.EN_ROUTE_PICKUP,
                    AssignmentStatus.ARRIVED_PICKUP,
                    AssignmentStatus.ONBOARD,
                    AssignmentStatus.EN_ROUTE_DROPOFF,
                ])
            )
            .order_by(Assignment.driver_id, Booking.scheduled_time)
            .all()
        )
        
        if not assignments:
            print("✅ Aucune assignation active trouvée")
            return
        
        # Grouper par chauffeur
        by_driver = {}
        for assignment in assignments:
            driver_id = assignment.driver_id
            if driver_id not in by_driver:
                by_driver[driver_id] = []
            by_driver[driver_id].append(assignment)
        
        conflicts_found = False
        
        # Vérifier chaque chauffeur
        for driver_id, driver_assignments in by_driver.items():
            driver = Driver.query.get(driver_id)
            driver_name = (
                getattr(driver.user, "full_name", None) 
                or getattr(driver, "name", None) 
                or f"Chauffeur #{driver_id}"
            )
            
            if len(driver_assignments) < 2:
                continue  # Pas de conflit possible avec une seule course
            
            print(f"👤 {driver_name} (#{driver_id}) - {len(driver_assignments)} course(s):")
            
            # Trier par scheduled_time
            sorted_assignments = sorted(
                driver_assignments,
                key=lambda a: a.booking.scheduled_time or datetime.min
            )
            
            # Vérifier chaque paire consécutive
            for i in range(len(sorted_assignments) - 1):
                current = sorted_assignments[i]
                next_assign = sorted_assignments[i + 1]
                
                current_booking = current.booking
                next_booking = next_assign.booking
                
                if not current_booking.scheduled_time or not next_booking.scheduled_time:
                    continue
                
                # Calculer le temps de trajet de la course actuelle
                current_pickup_lat = getattr(current_booking, "pickup_lat", None)
                current_pickup_lon = getattr(current_booking, "pickup_lon", None)
                current_dropoff_lat = getattr(current_booking, "dropoff_lat", None)
                current_dropoff_lon = getattr(current_booking, "dropoff_lon", None)
                
                # Calculer le temps de transition
                next_pickup_lat = getattr(next_booking, "pickup_lat", None)
                next_pickup_lon = getattr(next_booking, "pickup_lon", None)
                
                # Temps de trajet course actuelle (pickup → dropoff)
                if current_pickup_lat and current_pickup_lon and current_dropoff_lat and current_dropoff_lon:
                    trip_time_min = haversine_minutes(
                        (current_pickup_lat, current_pickup_lon),
                        (current_dropoff_lat, current_dropoff_lon),
                        avg_kmh=25
                    )
                else:
                    trip_time_min = 20  # Estimation par défaut
                
                # Temps de transition (dropoff actuel → pickup suivant)
                if current_dropoff_lat and current_dropoff_lon and next_pickup_lat and next_pickup_lon:
                    transition_time_min = haversine_minutes(
                        (current_dropoff_lat, current_dropoff_lon),
                        (next_pickup_lat, next_pickup_lon),
                        avg_kmh=25
                    )
                else:
                    transition_time_min = 15  # Estimation par défaut
                
                # Temps total nécessaire
                total_time_needed = (
                    trip_time_min +
                    dropoff_service_min +
                    transition_time_min +
                    pickup_service_min +
                    min_transition_margin_min
                )
                
                # Heure de fin estimée de la course actuelle
                current_end_time = current_booking.scheduled_time + timedelta(
                    minutes=trip_time_min + pickup_service_min + dropoff_service_min
                )
                
                # Heure de début nécessaire pour la course suivante
                required_start_time = next_booking.scheduled_time - timedelta(
                    minutes=transition_time_min + pickup_service_min + min_transition_margin_min
                )
                
                # Vérifier le conflit
                if current_end_time > required_start_time:
                    time_gap = (required_start_time - current_end_time).total_seconds() / 60
                    conflicts_found = True
                    print("   ⚠️  CONFLIT TEMPOREL détecté:")
                    print(f"      Course #{current_booking.id} ({current_booking.scheduled_time.strftime('%H:%M')})")
                    print(f"         → Fin estimée: {current_end_time.strftime('%H:%M')}")
                    print(f"      Course #{next_booking.id} ({next_booking.scheduled_time.strftime('%H:%M')})")
                    print(f"         → Début nécessaire: {required_start_time.strftime('%H:%M')}")
                    print(f"      ⏱️  Temps nécessaire: {total_time_needed}min")
                    print(f"      ⏱️  Écart disponible: {time_gap:.1f}min (NÉGATIF = CONFLIT)")
                    print(f"      📍 Course #{current_booking.id}: {getattr(current_booking, 'pickup_address', 'N/A')} → {getattr(current_booking, 'dropoff_address', 'N/A')}")
                    print(f"      📍 Course #{next_booking.id}: {getattr(next_booking, 'pickup_address', 'N/A')} → {getattr(next_booking, 'dropoff_address', 'N/A')}")
                    print()
                else:
                    time_gap = (required_start_time - current_end_time).total_seconds() / 60
                    print(f"   ✅ Course #{current_booking.id} ({current_booking.scheduled_time.strftime('%H:%M')}) → Course #{next_booking.id} ({next_booking.scheduled_time.strftime('%H:%M')})")
                    print(f"      Écart disponible: {time_gap:.1f}min (OK)")
            
            print()
        
        if not conflicts_found:
            print("✅ Aucun conflit temporel détecté !")
        else:
            print("❌ Conflits temporels détectés - action requise")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_temporal_conflicts.py <company_id>")
        sys.exit(1)
    
    company_id = int(sys.argv[1])
    check_temporal_conflicts(company_id)

