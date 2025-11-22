#!/usr/bin/env python3
"""Script pour vérifier que tous les paramètres configurés sont bien utilisés."""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Company
from services.unified_dispatch.settings import for_company


def verify_all_settings(company_id: int = 1):
    """Vérifie que tous les paramètres configurés sont bien utilisés."""
    print(f"🔍 Vérification de tous les paramètres pour company_id={company_id}\n")

    # Récupérer la company
    company = Company.query.get(company_id)
    if not company:
        print(f"❌ Company {company_id} non trouvée")
        return

    # Récupérer les paramètres depuis la DB
    autonomous_config = company.get_autonomous_config()
    dispatch_overrides = autonomous_config.get("dispatch_overrides", {})

    # Récupérer les settings calculés
    settings = for_company(company)

    print("=" * 80)
    print("📋 VÉRIFICATION DES PARAMÈTRES")
    print("=" * 80)

    # 1. Poids Heuristique
    print("\n1️⃣ POIDS HEURISTIQUE")
    print("-" * 80)
    heuristic_db = dispatch_overrides.get("heuristic", {})
    proximity_db = heuristic_db.get("proximity")
    driver_load_db = heuristic_db.get("driver_load_balance")
    priority_db = heuristic_db.get("priority")

    proximity_settings = getattr(settings.heuristic, "proximity", None)
    driver_load_settings = getattr(settings.heuristic, "driver_load_balance", None)
    priority_settings = getattr(settings.heuristic, "priority", None)

    print("   Proximité:")
    print(
        f"     DB: {proximity_db} → Settings: {proximity_settings} {'✅' if proximity_db == proximity_settings or (proximity_db is None and proximity_settings == 0.2) else '❌'}"
    )
    print("   Équilibre charge:")
    print(
        f"     DB: {driver_load_db} → Settings: {driver_load_settings} {'✅' if driver_load_db == driver_load_settings or (driver_load_db is None and driver_load_settings == 0.7) else '❌'}"
    )
    print("   Priorité:")
    print(
        f"     DB: {priority_db} → Settings: {priority_settings} {'✅' if priority_db == priority_settings or (priority_db is None and priority_settings == 0.06) else '❌'}"
    )

    # 2. Optimiseur (OR-Tools)
    print("\n2️⃣ OPTIMISEUR (OR-TOOLS)")
    print("-" * 80)
    solver_db = dispatch_overrides.get("solver", {})
    time_limit_db = solver_db.get("time_limit_sec")
    max_bookings_db = solver_db.get("max_bookings_per_driver")
    unassigned_penalty_db = solver_db.get("unassigned_penalty_base")

    time_limit_settings = getattr(settings.solver, "time_limit_sec", None)
    max_bookings_settings = getattr(settings.solver, "max_bookings_per_driver", None)
    unassigned_penalty_settings = getattr(settings.solver, "unassigned_penalty_base", None)

    print("   Temps limite (secondes):")
    print(
        f"     DB: {time_limit_db} → Settings: {time_limit_settings} {'✅' if time_limit_db == time_limit_settings or (time_limit_db is None and time_limit_settings == 60) else '❌'}"
    )
    print("   Courses max par chauffeur:")
    print(
        f"     DB: {max_bookings_db} → Settings: {max_bookings_settings} {'✅' if max_bookings_db == max_bookings_settings or (max_bookings_db is None and max_bookings_settings == 6) else '❌'}"
    )
    print("   Pénalité non-assigné:")
    print(
        f"     DB: {unassigned_penalty_db} → Settings: {unassigned_penalty_settings} {'✅' if unassigned_penalty_db == unassigned_penalty_settings or (unassigned_penalty_db is None and unassigned_penalty_settings == 10000) else '❌'}"
    )

    # 3. Temps de Service
    print("\n3️⃣ TEMPS DE SERVICE")
    print("-" * 80)
    service_times_db = dispatch_overrides.get("service_times", {})
    pickup_db = service_times_db.get("pickup_service_min")
    dropoff_db = service_times_db.get("dropoff_service_min")
    margin_db = service_times_db.get("min_transition_margin_min")

    pickup_settings = getattr(settings.service_times, "pickup_service_min", None)
    dropoff_settings = getattr(settings.service_times, "dropoff_service_min", None)
    margin_settings = getattr(settings.service_times, "min_transition_margin_min", None)

    print("   Pickup (minutes):")
    print(
        f"     DB: {pickup_db} → Settings: {pickup_settings} {'✅' if pickup_db == pickup_settings or (pickup_db is None and pickup_settings == 5) else '❌'}"
    )
    print("   Dropoff (minutes):")
    print(
        f"     DB: {dropoff_db} → Settings: {dropoff_settings} {'✅' if dropoff_db == dropoff_settings or (dropoff_db is None and dropoff_settings == 10) else '❌'}"
    )
    print("   Marge transition (minutes):")
    print(
        f"     DB: {margin_db} → Settings: {margin_settings} {'✅' if margin_db == margin_settings or (margin_db is None and margin_settings == 15) else '❌'}"
    )

    # 4. Regroupement de Courses
    print("\n4️⃣ REGROUPEMENT DE COURSES")
    print("-" * 80)
    pooling_db = dispatch_overrides.get("pooling", {})
    pooling_enabled_db = pooling_db.get("enabled")

    pooling_enabled_settings = getattr(settings.pooling, "enabled", None)

    print("   Activer le regroupement:")
    print(
        f"     DB: {pooling_enabled_db} → Settings: {pooling_enabled_settings} {'✅' if pooling_enabled_db == pooling_enabled_settings or (pooling_enabled_db is None and pooling_enabled_settings) else '❌'}"
    )

    # 5. Équité Chauffeurs
    print("\n5️⃣ ÉQUITÉ CHAUFFEURS")
    print("-" * 80)
    fairness_db = dispatch_overrides.get("fairness", {})
    fairness_enabled_db = fairness_db.get("enable_fairness")
    fairness_window_db = fairness_db.get("fairness_window_days")
    fairness_weight_db = fairness_db.get("fairness_weight")

    fairness_enabled_settings = getattr(settings.fairness, "enable_fairness", None)
    fairness_window_settings = getattr(settings.fairness, "fairness_window_days", None)
    fairness_weight_settings = getattr(settings.fairness, "fairness_weight", None)

    print("   Activer l'équité:")
    print(
        f"     DB: {fairness_enabled_db} → Settings: {fairness_enabled_settings} {'✅' if fairness_enabled_db == fairness_enabled_settings or (fairness_enabled_db is None and fairness_enabled_settings) else '❌'}"
    )
    print("   Fenêtre d'équité (jours):")
    print(
        f"     DB: {fairness_window_db} → Settings: {fairness_window_settings} {'✅' if fairness_window_db == fairness_window_settings or (fairness_window_db is None and fairness_window_settings == 7) else '❌'}"
    )
    print("   Poids équité (0-1):")
    print(
        f"     DB: {fairness_weight_db} → Settings: {fairness_weight_settings} {'✅' if fairness_weight_db == fairness_weight_settings or (fairness_weight_db is None and fairness_weight_settings == 0.3) else '❌'}"
    )

    # 6. Chauffeur d'Urgence
    print("\n6️⃣ CHAUFFEUR D'URGENCE")
    print("-" * 80)
    emergency_db = dispatch_overrides.get("emergency", {})
    allow_emergency_db = emergency_db.get("allow_emergency_drivers")
    emergency_penalty_db = emergency_db.get("emergency_penalty") or emergency_db.get("emergency_per_stop_penalty")

    allow_emergency_settings = getattr(settings.emergency, "allow_emergency_drivers", None)
    emergency_penalty_settings = getattr(settings.emergency, "emergency_penalty", None)

    print("   Autoriser chauffeurs d'urgence:")
    print(
        f"     DB: {allow_emergency_db} → Settings: {allow_emergency_settings} {'✅' if allow_emergency_db == allow_emergency_settings or (allow_emergency_db is None and allow_emergency_settings) else '❌'}"
    )
    print("   Pénalité d'utilisation (0-1000):")
    print(
        f"     DB: {emergency_penalty_db} → Settings: {emergency_penalty_settings} {'✅' if emergency_penalty_db == emergency_penalty_settings or (emergency_penalty_db is None and emergency_penalty_settings == 900.0) else '❌'}"
    )
    if emergency_penalty_settings:
        malus = -(emergency_penalty_settings / 180.0)
        print(f"     → Malus appliqué: {malus:.3f}")

    # 7. Préférence Chauffeur (dans overrides, pas dans settings)
    print("\n7️⃣ PRÉFÉRENCE CHAUFFEUR")
    print("-" * 80)
    preferred_driver_db = dispatch_overrides.get("preferred_driver_id")
    driver_load_multipliers_db = dispatch_overrides.get("driver_load_multipliers", {})

    print("   Chauffeur préféré:")
    print(f"     DB: {preferred_driver_db} {'✅' if preferred_driver_db else '⚠️  Non configuré'}")
    if preferred_driver_db and preferred_driver_db in driver_load_multipliers_db:
        multiplier = driver_load_multipliers_db[preferred_driver_db]
        print("   Multiplicateur de charge:")
        print(f"     DB: {multiplier} {'✅' if multiplier else '⚠️  Non configuré'}")

    # Résumé
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ")
    print("=" * 80)
    print("✅ Paramètres vérifiés depuis: autonomous_config.dispatch_overrides")
    print("✅ Settings calculés via: for_company(company)")
    print("\n💡 Note: Les valeurs par défaut sont utilisées si non configurées dans la DB")

    print("\n" + "=" * 80)
    print("✅ Vérification terminée")
    print("=" * 80)


if __name__ == "__main__":
    # Utiliser Flask app context
    from app import create_app

    app = create_app()

    with app.app_context():
        company_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
        verify_all_settings(company_id)
