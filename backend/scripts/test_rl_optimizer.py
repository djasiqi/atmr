#!/usr/bin/env python3
# ruff: noqa: T201, DTZ001
"""
Script de test de l'optimiseur RL sur le dispatch du 22 octobre.

Compare les résultats :
- Avant : Assignations heuristiques
- Après : Assignations optimisées par RL
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import create_app
from models import Booking, Driver
from services.unified_dispatch.rl_optimizer import RLDispatchOptimizer

# Créer l'app Flask pour accès à la DB
app = create_app()

with app.app_context():
    print("=" * 80)
    print("🧪 TEST DE L'OPTIMISEUR RL")
    print("=" * 80)
    print()

    # Récupérer les bookings du 22 octobre
    date = datetime(2025, 10, 22).date()
    bookings = (
        Booking.query.filter(
            Booking.scheduled_time >= datetime.combine(date, datetime.min.time()),
            Booking.scheduled_time < datetime.combine(date, datetime.max.time()),
            Booking.driver_id.isnot(None),  # Seulement les assignés  # pyright: ignore[reportAttributeAccessIssue]
        )
        .order_by(Booking.scheduled_time)
        .all()
    )

    if not bookings:
        print("❌ Aucune course assignée trouvée pour le 22 octobre")
        sys.exit(1)

    print(f"📦 {len(bookings)} courses chargées")

    # Récupérer les drivers impliqués
    driver_ids = {b.driver_id for b in bookings}
    drivers = Driver.query.filter(Driver.id.in_(driver_ids)).all()
    print(f"👥 {len(drivers)} chauffeurs impliqués")
    print()

    # Construire les assignations initiales (heuristique)
    initial_assignments = [
        {"booking_id": b.id, "driver_id": b.driver_id} for b in bookings
    ]

    # Calculer la répartition initiale
    driver_loads_initial = {}
    for a in initial_assignments:
        did = a["driver_id"]
        driver_loads_initial[did] = driver_loads_initial.get(did, 0) + 1

    driver_names = {d.id: f"{d.user.first_name} {d.user.last_name}" for d in drivers}

    print("📊 AVANT (Heuristique) :")
    print("-" * 80)
    for driver_id, count in sorted(
        driver_loads_initial.items(), key=lambda x: x[1], reverse=True
    ):
        driver_name = driver_names.get(driver_id, f"Driver {driver_id}")
        bar = "█" * count
        print(f"  {driver_name:20} : {count:2d} courses {bar}")

    max_load_initial = max(driver_loads_initial.values())
    min_load_initial = min(driver_loads_initial.values())
    gap_initial = max_load_initial - min_load_initial
    print()
    print(f"  ÉCART : {gap_initial} courses (max={max_load_initial}, min={min_load_initial})")
    print()

    # Créer l'optimiseur RL
    print("🧠 OPTIMISATION RL EN COURS...")
    print("-" * 80)

    optimizer = RLDispatchOptimizer(
        model_path="data/rl/models/dispatch_optimized_v1.pth",
        max_swaps=10,
        min_improvement=0.5,
    )

    if not optimizer.is_available():
        print("❌ Modèle RL non disponible")
        sys.exit(1)

    # Optimiser
    optimized_assignments = optimizer.optimize_assignments(
        initial_assignments=initial_assignments,
        bookings=bookings,
        drivers=drivers,
    )

    # Calculer la nouvelle répartition
    driver_loads_optimized = {}
    for a in optimized_assignments:
        did = a["driver_id"]
        driver_loads_optimized[did] = driver_loads_optimized.get(did, 0) + 1

    print()
    print("=" * 80)
    print("📊 APRÈS (Heuristique + RL) :")
    print("-" * 80)
    for driver_id, count in sorted(
        driver_loads_optimized.items(), key=lambda x: x[1], reverse=True
    ):
        driver_name = driver_names.get(driver_id, f"Driver {driver_id}")
        bar = "█" * count
        old_count = driver_loads_initial.get(driver_id, 0)
        delta = count - old_count
        delta_str = f"({delta:+d})" if delta != 0 else ""
        print(f"  {driver_name:20} : {count:2d} courses {bar} {delta_str}")

    max_load_optimized = max(driver_loads_optimized.values())
    min_load_optimized = min(driver_loads_optimized.values())
    gap_optimized = max_load_optimized - min_load_optimized
    print()
    print(
        f"  ÉCART : {gap_optimized} courses (max={max_load_optimized}, min={min_load_optimized})"
    )
    print()

    # Rapport final
    print("=" * 80)
    print("📈 AMÉLIORATION :")
    print("-" * 80)
    improvement = gap_initial - gap_optimized
    improvement_pct = (improvement / gap_initial * 100) if gap_initial > 0 else 0

    print(f"  Écart initial    : {gap_initial} courses")
    print(f"  Écart optimisé   : {gap_optimized} courses")
    print(f"  Amélioration     : {improvement} courses ({improvement_pct:.1f}%)")
    print()

    if gap_optimized <= 1:
        print("  🎯 OBJECTIF ATTEINT : Écart ≤ 1 course !")
    elif improvement > 0:
        print(f"  ✅ AMÉLIORATION : Écart réduit de {improvement_pct:.0f}% !")
    else:
        print("  ⚠️  Pas d'amélioration possible avec le modèle actuel")

    print()

    # Détails des réassignations
    changes = []
    for initial, optimized in zip(initial_assignments, optimized_assignments, strict=False):
        if initial["driver_id"] != optimized["driver_id"]:
            booking = next(b for b in bookings if b.id == initial["booking_id"])
            old_driver = driver_names.get(
                initial["driver_id"], f"Driver {initial['driver_id']}"
            )
            new_driver = driver_names.get(
                optimized["driver_id"], f"Driver {optimized['driver_id']}"
            )
            # Récupérer le nom du client
            client_name = "Course"
            if hasattr(booking, "client") and booking.client:
                client_name = f"{booking.client.first_name} {booking.client.last_name}"
            elif hasattr(booking, "client_name") and booking.client_name:
                client_name = booking.client_name

            changes.append(
                {
                    "booking_id": booking.id,
                    "time": booking.scheduled_time.strftime("%H:%M"),
                    "client": client_name,
                    "old_driver": old_driver,
                    "new_driver": new_driver,
                }
            )

    if changes:
        print("🔄 RÉASSIGNATIONS EFFECTUÉES :")
        print("-" * 80)
        for change in changes:
            print(
                f"  {change['time']} - {change['client']:30} : {change['old_driver']} → {change['new_driver']}"
            )
        print()
    else:
        print("  Aucune réassignation nécessaire")
        print()

    print("=" * 80)

