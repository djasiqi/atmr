#!/usr/bin/env python3
# ruff: noqa: T201, W293
"""
Script de monitoring pour l'entraînement RL avec données nettoyées
"""

import json
from pathlib import Path


def monitor_rl_training():
    """Surveille l'entraînement RL en cours"""

    print("=" * 80)
    print("📊 MONITORING ENTRAÎNEMENT RL - DONNÉES NETTOYÉES")
    print("=" * 80)

    # Vérifier les fichiers de données
    cleaned_data_file = Path("/app/data/rl/historical_dispatches_cleaned.json")
    model_file = Path("/app/data/rl/models/dispatch_optimized_v3_cleaned.pth")

    print(f"📂 Données RL : {'✅' if cleaned_data_file.exists() else '❌'} {cleaned_data_file}")
    print(f"🧠 Modèle RL : {'✅' if model_file.exists() else '⏳ En cours...'} {model_file}")

    if cleaned_data_file.exists():
        with open(cleaned_data_file, encoding='utf-8') as f:
            data = json.load(f)

        dispatches = data['dispatches']
        metadata = data['metadata']

        print("\n📊 DONNÉES D'ENTRAÎNEMENT :")
        print(f"   - Dispatches : {len(dispatches)}")
        print(f"   - Total bookings : {metadata['total_bookings']}")
        print(f"   - Total drivers : {metadata['total_drivers']}")
        print(f"   - Période : {metadata['date_range']['start']} → {metadata['date_range']['end']}")

        # Analyser les conducteurs
        all_drivers = set()
        for dispatch in dispatches:
            all_drivers.update(dispatch['driver_names'])

        print("\n👥 CONDUCTEURS IDENTIFIÉS :")
        for driver in sorted(all_drivers):
            print(f"   - {driver}")

        # Statistiques par dispatch
        bookings_per_dispatch = [d['num_bookings'] for d in dispatches]
        drivers_per_dispatch = [d['num_drivers'] for d in dispatches]

        print("\n📈 STATISTIQUES DES DISPATCHES :")
        print(f"   - Bookings/dispatch : {min(bookings_per_dispatch)}-{max(bookings_per_dispatch)} (moy: {sum(bookings_per_dispatch)/len(bookings_per_dispatch):.1f})")
        print(f"   - Drivers/dispatch : {min(drivers_per_dispatch)}-{max(drivers_per_dispatch)} (moy: {sum(drivers_per_dispatch)/len(drivers_per_dispatch):.1f})")

    print("\n🎯 OBJECTIFS DE L'ENTRAÎNEMENT :")
    print("   - Épisodes : 15,000")
    print("   - Durée estimée : 4-6 heures")
    print("   - Amélioration attendue : -50% d'écart entre conducteurs")
    print("   - Modèle final : dispatch_optimized_v3_cleaned.pth")

    print("\n⏳ STATUT ACTUEL :")
    if model_file.exists():
        size_mb = model_file.stat().st_size / (1024 * 1024)
        print(f"   ✅ Modèle créé ({size_mb:.1f} MB)")
        print("   🚀 Prêt pour déploiement")
    else:
        print("   🔄 Entraînement en cours...")
        print("   📊 Surveillez les logs pour le progrès")

    print("\n🔍 POUR SUIVRE L'ENTRAÎNEMENT :")
    print("   docker-compose logs -f api | grep 'Episode'")
    print("   docker-compose exec api ls -la /app/data/rl/models/")

    return {
        'dispatches_ready': cleaned_data_file.exists(),
        'model_ready': model_file.exists(),
        'total_dispatches': len(dispatches) if cleaned_data_file.exists() else 0,
        'total_bookings': metadata['total_bookings'] if cleaned_data_file.exists() else 0
    }


if __name__ == "__main__":
    monitor_rl_training()
