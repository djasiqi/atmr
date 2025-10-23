#!/usr/bin/env python3
"""
Script de test pour vérifier l'intégration RL dans le système de dispatch.
"""

import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent))

from services.unified_dispatch.rl_optimizer import RLDispatchOptimizer


def test_rl_integration():
    """Test l'intégration RL."""

    print("=================================================================================")  # noqa: T201
    print("🧪 TEST D'INTÉGRATION RL")  # noqa: T201
    print("=================================================================================")  # noqa: T201

    # 1. Test de chargement du modèle
    print("\n📦 Test de chargement du modèle...")  # noqa: T201
    optimizer = RLDispatchOptimizer()

    if optimizer.is_available():
        print("✅ Modèle RL chargé avec succès")  # noqa: T201
        print(f"   - Chemin: {optimizer.model_path}")  # noqa: T201
        print(f"   - Agent disponible: {optimizer.agent is not None}")  # noqa: T201
        print(f"   - Environnement disponible: {optimizer.env is not None}")  # noqa: T201
    else:
        print("❌ Modèle RL non disponible")  # noqa: T201
        print(f"   - Chemin: {optimizer.model_path}")  # noqa: T201
        print(f"   - Fichier existe: {optimizer.model_path.exists()}")  # noqa: T201
        return False

    # 2. Test avec des données simulées
    print("\n🧪 Test avec données simulées...")  # noqa: T201

    # Créer des données de test
    class MockBooking:
        def __init__(self, id, pickup_lat=46.2044, pickup_lon=6.1432, dropoff_lat=46.2044, dropoff_lon=6.1432):
            self.id = id
            self.pickup_lat = pickup_lat
            self.pickup_lon = pickup_lon
            self.dropoff_lat = dropoff_lat
            self.dropoff_lon = dropoff_lon

    class MockDriver:
        def __init__(self, id, latitude=46.2044, longitude=6.1432):
            self.id = id
            self.latitude = latitude
            self.longitude = longitude

    # Données de test
    bookings = [
        MockBooking(1, 46.2044, 6.1432, 46.2100, 6.1500),
        MockBooking(2, 46.2044, 6.1432, 46.2200, 6.1600),
        MockBooking(3, 46.2044, 6.1432, 46.2300, 6.1700),
        MockBooking(4, 46.2044, 6.1432, 46.2400, 6.1800),
        MockBooking(5, 46.2044, 6.1432, 46.2500, 6.1900),
    ]

    drivers = [
        MockDriver(1, 46.2044, 6.1432),
        MockDriver(2, 46.2044, 6.1432),
        MockDriver(3, 46.2044, 6.1432),
    ]

    # Assignations initiales (déséquilibrées)
    initial_assignments = [
        {"booking_id": 1, "driver_id": 1},
        {"booking_id": 2, "driver_id": 1},
        {"booking_id": 3, "driver_id": 1},
        {"booking_id": 4, "driver_id": 2},
        {"booking_id": 5, "driver_id": 2},
    ]

    print(f"   - {len(bookings)} bookings")  # noqa: T201
    print(f"   - {len(drivers)} drivers")  # noqa: T201
    print(f"   - Écart initial: {optimizer._calculate_gap(initial_assignments, drivers)} courses")  # noqa: T201

    # 3. Test d'optimisation
    print("\n🧠 Test d'optimisation...")  # noqa: T201

    try:
        optimized_assignments = optimizer.optimize_assignments(
            initial_assignments=initial_assignments,
            bookings=bookings,
            drivers=drivers
        )

        final_gap = optimizer._calculate_gap(optimized_assignments, drivers)
        initial_gap = optimizer._calculate_gap(initial_assignments, drivers)

        print("✅ Optimisation terminée")  # noqa: T201
        print(f"   - Écart initial: {initial_gap} courses")  # noqa: T201
        print(f"   - Écart final: {final_gap} courses")  # noqa: T201
        print(f"   - Amélioration: {initial_gap - final_gap} courses")  # noqa: T201

        if final_gap < initial_gap:
            print("🎉 Optimisation réussie !")  # noqa: T201
        else:
            print("⚠️  Pas d'amélioration détectée")  # noqa: T201

        # Afficher les assignations finales
        print("\n📋 Assignations finales:")  # noqa: T201
        for assignment in optimized_assignments:
            print(f"   - Booking {assignment['booking_id']} → Driver {assignment['driver_id']}")  # noqa: T201

    except Exception as e:
        print(f"❌ Erreur lors de l'optimisation: {e}")  # noqa: T201
        import traceback
        traceback.print_exc()
        return False

    print("\n🎉 Test d'intégration RL réussi !")  # noqa: T201
    return True

if __name__ == "__main__":
    success = test_rl_integration()
    sys.exit(0 if success else 1)
