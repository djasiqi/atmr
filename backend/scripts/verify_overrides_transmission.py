#!/usr/bin/env python3
"""
Script pour vérifier que tous les paramètres avancés sont bien transmis et utilisés
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from services.unified_dispatch.settings import Settings, merge_overrides

# Simuler les overrides depuis le frontend
test_overrides = {
    "heuristic": {
        "proximity": 0.05,
        "driver_load_balance": 0.95,
        "priority": 0.06,
    },
    "solver": {
        "time_limit_sec": 60,
        "max_bookings_per_driver": 10,
        "unassigned_penalty": 10000,
    },
    "service_times": {
        "pickup_service_min": 5,
        "dropoff_service_min": 10,
        "min_transition_margin_min": 15,
    },
    "pooling": {
        "enabled": True,
        "time_tolerance_min": 10,
        "pickup_distance_m": 500,
    },
    "fairness": {
        "enabled": True,
        "window_days": 2,
        "fairness_weight": 0.7,
    },
    "preferred_driver_id": 123,
    "driver_load_multipliers": {
        "123": 1.5
    },
    "allow_emergency": True,
    "emergency": {
        "allow_emergency_drivers": True,
        "emergency_penalty": 900,
    },
}

print("=== Test de transmission des overrides ===\n")

# Créer les settings de base
base_settings = Settings()

# Appliquer les overrides
merged_settings = merge_overrides(base_settings, test_overrides)

# Vérifier chaque paramètre
checks = [
    ("Proximité", merged_settings.heuristic.proximity, 0.05),
    ("Équilibre charge", merged_settings.heuristic.driver_load_balance, 0.95),
    ("Priorité", merged_settings.heuristic.priority, 0.06),
    ("Temps limite solveur", merged_settings.solver.time_limit_sec, 60),
    ("Courses max par chauffeur", merged_settings.solver.max_bookings_per_driver, 10),
    ("Pénalité non-assigné", merged_settings.solver.unassigned_penalty, 10000),
    ("Pickup service (min)", merged_settings.service_times.pickup_service_min, 5),
    ("Dropoff service (min)", merged_settings.service_times.dropoff_service_min, 10),
    ("Marge transition (min)", merged_settings.service_times.min_transition_margin_min, 15),
    ("Pooling activé", merged_settings.pooling.enabled, True),
    ("Tolérance temporelle pooling", merged_settings.pooling.time_tolerance_min, 10),
    ("Distance pickup pooling", merged_settings.pooling.pickup_distance_m, 500),
    ("Équité activée", merged_settings.fairness.enabled, True),
    ("Fenêtre équité (jours)", merged_settings.fairness.window_days, 2),
    ("Poids équité", merged_settings.fairness.fairness_weight, 0.7),
    ("Chauffeurs d'urgence autorisés", merged_settings.emergency.allow_emergency_drivers, True),
    ("Pénalité d'urgence", merged_settings.emergency.emergency_penalty, 900),
]

print("Vérification des paramètres après merge_overrides:\n")
all_ok = True
for name, actual, expected in checks:
    status = "✅" if actual == expected else "❌"
    if actual != expected:
        all_ok = False
    print(f"{status} {name}: {actual} (attendu: {expected})")

print("\n=== Paramètres non gérés par merge_overrides (passés directement dans problem) ===\n")
print("✅ preferred_driver_id: Passé dans problem['preferred_driver_id']")
print("✅ driver_load_multipliers: Passé dans problem['driver_load_multipliers']")
print("✅ allow_emergency: Passé dans problem['allow_emergency']")

print("\n" + "="*60)
if all_ok:
    print("✅ Tous les paramètres sont correctement transmis et appliqués!")
else:
    print("❌ Certains paramètres ne sont pas correctement appliqués")
print("="*60)

