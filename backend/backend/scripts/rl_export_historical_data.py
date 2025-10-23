#!/usr/bin/env python3
# ruff: noqa: T201
"""
Script d'export des données historiques de dispatch pour entraînement RL.

Extrait tous les dispatch_runs passés avec :
- Coordonnées GPS réelles (pickup/dropoff)
- Distances calculées (haversine)
- Temps de trajet (estimés ou réels si disponibles)
- Répartition par chauffeur (équité)
- Métriques de qualité

Auteur: ATMR Project
Date: 21 octobre 2025
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from models import Assignment, DispatchRun, Driver
from models.enums import DispatchStatus
from shared.geo_utils import haversine_distance


def export_historical_dispatches(
    company_id: int = 1,
    start_date: str = "2025-01-01",
    end_date: str = "2025-10-21",
    output_file: str = "data/rl/historical_dispatches.json",
    min_bookings: int = 5,  # Skip dispatches avec < 5 courses
) -> None:
    """
    Exporte les dispatches historiques au format JSON pour entraînement RL.

    Args:
        company_id: ID de la compagnie
        start_date: Date de début (YYYY-MM-DD)
        end_date: Date de fin (YYYY-MM-DD)
        output_file: Chemin du fichier de sortie
        min_bookings: Nombre minimum de bookings par dispatch
    """
    print("=" * 80)
    print("🧠 EXPORT DES DONNÉES HISTORIQUES POUR ENTRAÎNEMENT RL")
    print("=" * 80)
    print(f"📅 Période : {start_date} → {end_date}")
    print(f"🏢 Company ID : {company_id}")
    print(f"📊 Min bookings par dispatch : {min_bookings}")
    print()

    # Récupérer tous les dispatch_runs de la période
    runs = (
        DispatchRun.query.filter(
            DispatchRun.company_id == company_id,
            DispatchRun.day >= datetime.fromisoformat(start_date).date(),
            DispatchRun.day <= datetime.fromisoformat(end_date).date(),
            DispatchRun.status == DispatchStatus.COMPLETED,
        )
        .order_by(DispatchRun.day.desc())
        .all()
    )

    print(f"📊 Récupération de {len(runs)} dispatch runs...")
    print()

    dispatches = []
    skipped_count = 0
    total_bookings = 0
    total_load_gap = 0

    for idx, run in enumerate(runs):
        # Récupérer les bookings et assignments
        assignments = Assignment.query.filter_by(dispatch_run_id=run.id).all()

        if len(assignments) < min_bookings:
            skipped_count += 1
            continue  # Skip runs avec peu d'assignments

        # Récupérer les drivers impliqués
        driver_ids = {a.driver_id for a in assignments if a.driver_id}
        drivers_data = Driver.query.filter(Driver.id.in_(driver_ids)).all()

        # Calculer les métriques
        driver_loads = {}
        total_distance = 0.0
        retards = 0
        bookings_data = []

        for a in assignments:
            driver_id = a.driver_id
            if not driver_id:
                continue  # Skip unassigned

            driver_loads[driver_id] = driver_loads.get(driver_id, 0) + 1

            # Récupérer le booking
            booking = a.booking

            # Calculer distance (haversine si GPS disponible)
            distance_km = 0.0
            if booking.pickup_lat and booking.dropoff_lat:
                try:
                    distance_km = haversine_distance(
                        (float(booking.pickup_lat), float(booking.pickup_lon)),
                        (float(booking.dropoff_lat), float(booking.dropoff_lon)),
                    )
                    total_distance += distance_km
                except Exception:
                    pass

            # Calculer temps de trajet estimé (distance / vitesse moyenne)
            avg_speed_kmh = 30.0  # Vitesse moyenne en ville
            estimated_duration_minutes = (distance_km / avg_speed_kmh) * 60 if distance_km > 0 else 0

            # Détecter retards (si disponible)
            actual_delay = 0
            if hasattr(a, "actual_pickup_time") and a.actual_pickup_time and booking.scheduled_time:
                try:
                    delay = (a.actual_pickup_time - booking.scheduled_time).total_seconds() / 60
                    actual_delay = int(delay)
                    if delay > 5:
                        retards += 1
                except Exception:
                    pass

            # Export booking data
            bookings_data.append(
                {
                    "id": a.booking_id,
                    "scheduled_time": booking.scheduled_time.isoformat() if booking.scheduled_time else None,
                    "pickup_lat": float(booking.pickup_lat) if booking.pickup_lat else None,
                    "pickup_lon": float(booking.pickup_lon) if booking.pickup_lon else None,
                    "dropoff_lat": float(booking.dropoff_lat) if booking.dropoff_lat else None,
                    "dropoff_lon": float(booking.dropoff_lon) if booking.dropoff_lon else None,
                    "pickup_location": booking.pickup_location,
                    "dropoff_location": booking.dropoff_location,
                    "distance_km": round(distance_km, 2),
                    "estimated_duration_minutes": round(estimated_duration_minutes, 1),
                    "assigned_driver_id": a.driver_id,
                    "actual_delay_minutes": actual_delay,
                }
            )

        # Calculer écart de charge (équité)
        if driver_loads:
            max_load = max(driver_loads.values())
            min_load = min(driver_loads.values())
            load_gap = max_load - min_load
            total_load_gap += load_gap
        else:
            load_gap = 0

        # Calculer score global (métrique qualité)
        quality_score = max(
            0,
            100
            - (load_gap * 15)  # Pénalité équité (15 points par course d'écart)
            - (total_distance * 0.3)  # Pénalité distance (0.3 points par km)
            - (retards * 8),  # Pénalité retards (8 points par retard)
        )

        # Export dispatch
        dispatch_data = {
            "id": run.id,
            "date": run.day.isoformat(),
            "num_bookings": len(assignments),
            "num_drivers": len(driver_loads),
            "driver_loads": driver_loads,
            "load_gap": load_gap,
            "total_distance_km": round(total_distance, 2),
            "avg_distance_per_booking": round(total_distance / len(assignments), 2) if assignments else 0,
            "retards_count": retards,
            "quality_score": round(quality_score, 2),
            "bookings": bookings_data,
            "drivers": [
                {
                    "id": d.id,
                    "name": f"{getattr(d.user, 'first_name', '')} {getattr(d.user, 'last_name', '')}".strip() or f"Driver {d.id}",
                    "is_emergency": getattr(d, "is_emergency", False),
                    "num_assignments": driver_loads.get(d.id, 0),
                }
                for d in drivers_data
            ],
        }

        dispatches.append(dispatch_data)
        total_bookings += len(assignments)

        # Progress log tous les 50 dispatches
        if (idx + 1) % 50 == 0:
            print(f"⏳ Traité {idx + 1}/{len(runs)} dispatches...")

    # Statistiques finales
    print()
    print("=" * 80)
    print("📊 STATISTIQUES D'EXPORT")
    print("=" * 80)
    print(f"✅ Dispatches exportés : {len(dispatches)}")
    print(f"⏭️  Dispatches skippés  : {skipped_count} (< {min_bookings} bookings)")
    print(f"📦 Total bookings      : {total_bookings}")
    print(f"📈 Avg bookings/dispatch : {total_bookings / len(dispatches):.1f}" if dispatches else "N/A")
    print()

    # Variables par défaut
    avg_gap = 0.0
    avg_score = 0.0
    avg_distance = 0.0
    gap_distribution = {}

    if dispatches:
        avg_gap = sum(d["load_gap"] for d in dispatches) / len(dispatches)
        avg_score = sum(d["quality_score"] for d in dispatches) / len(dispatches)
        avg_distance = sum(d["total_distance_km"] for d in dispatches) / len(dispatches)

        print(f"⚖️  Écart moyen         : {avg_gap:.2f} courses")
        print(f"🏆 Score qualité moyen : {avg_score:.1f}/100")
        print(f"📏 Distance moyenne    : {avg_distance:.1f} km/dispatch")
        print()

        # Distribution des écarts
        for d in dispatches:
            gap = d["load_gap"]
            gap_distribution[gap] = gap_distribution.get(gap, 0) + 1

        print("📊 Distribution des écarts :")
        for gap in sorted(gap_distribution.keys()):
            count = gap_distribution[gap]
            pct = (count / len(dispatches)) * 100
            bar = "█" * int(pct / 2)
            print(f"   Écart {gap}: {count:4d} ({pct:5.1f}%) {bar}")
        print()

    # Sauvegarder
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_data = {
        "company_id": company_id,
        "period": {"start": start_date, "end": end_date},
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "total_dispatches": len(dispatches),
        "total_bookings": total_bookings,
        "statistics": {
            "avg_load_gap": round(avg_gap, 2) if dispatches else 0,
            "avg_quality_score": round(avg_score, 2) if dispatches else 0,
            "avg_total_distance": round(avg_distance, 2) if dispatches else 0,
            "gap_distribution": gap_distribution,
        },
        "dispatches": dispatches,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Données exportées vers : {output_path.absolute()}")
    print(f"📦 Taille du fichier     : {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    print("🚀 Prochaine étape : Lancer l'entraînement RL !")
    print("   python backend/scripts/rl_train_offline.py")
    print()


if __name__ == "__main__":
    # Créer l'app Flask pour accès à la DB
    app = create_app()

    with app.app_context():
        export_historical_dispatches(
            company_id=1,
            start_date="2025-10-19",  # Ajuster selon vos données
            end_date="2025-10-22",  # Inclure le 20 et 21 octobre
            output_file="data/rl/historical_dispatches.json",
            min_bookings=1,  # Accept même les petits dispatches
        )

