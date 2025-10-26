#!/usr/bin/env python3
"""Script pour convertir les données Excel nettoyées en format RL
et lancer un entraînement avec les vraies données de conducteurs.
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.geo_utils import haversine_distance


def convert_cleaned_data_to_rl(
    cleaned_data_file: str = "/app/training_data_cleaned_final.json",
    output_file: str = "/app/data/rl/historical_dispatches_cleaned.json",
    min_courses_per_day: int = 3,
) -> None:
    """Convertit les données Excel nettoyées en format RL.
    
    Args:
        cleaned_data_file: Fichier JSON des données nettoyées
        output_file: Fichier de sortie pour RL
        min_courses_per_day: Minimum de courses par jour pour créer un dispatch

    """
    print("=" * 80)
    print("🔄 CONVERSION DONNÉES NETTOYÉES → FORMAT RL")
    print("=" * 80)
    print("📂 Source : {cleaned_data_file}")
    print("📤 Destination : {output_file}")
    print()

    # Charger les données nettoyées
    with Path(cleaned_data_file, encoding="utf-8").open() as f:
        data = json.load(f)

    training_data = data["training_data"]
    print("📊 {len(training_data)} enregistrements chargés")

    # Grouper par date pour créer des dispatches
    dispatches_by_date = {}

    for record in training_data:
        date_str = record["date"]
        if not date_str:
            continue

        if date_str not in dispatches_by_date:
            dispatches_by_date[date_str] = []

        dispatches_by_date[date_str].append(record)

    print("📅 {len(dispatches_by_date)} jours trouvés")

    # Créer les dispatches RL
    rl_dispatches = []

    for date_str, records in dispatches_by_date.items():
        if len(records) < min_courses_per_day:
            continue

        # Extraire les conducteurs uniques
        drivers = set()
        for record in records:
            if record["conducteur_aller_name"]:
                drivers.add(record["conducteur_aller_name"])
            if record["conducteur_retour_name"]:
                drivers.add(record["conducteur_retour_name"])

        drivers = list(drivers)
        if len(drivers) < 2:  # Au moins 2 conducteurs pour un dispatch
            continue

        # Créer les bookings (courses)
        bookings = []
        for record in records:
            if not record["adresse_depart"] or not record["adresse_arrivee"]:
                continue

            # Vérifier que les coordonnées sont disponibles
            if (record.get("geocoding_depart") and
                record["geocoding_depart"].get("status") == "OK" and
                record.get("geocoding_arrivee") and
                record["geocoding_arrivee"].get("status") == "OK"):

                booking = {
                    "id": f"booking_{len(bookings)}",
                    "pickup_lat": record["geocoding_depart"]["lat"],
                    "pickup_lng": record["geocoding_depart"]["lng"],
                    "dropoff_lat": record["geocoding_arrivee"]["lat"],
                    "dropoff_lng": record["geocoding_arrivee"]["lng"],
                    "pickup_time": record["heure_depart_aller"] or "09:00",
                    "dropoff_time": record["heure_depart_retour"] or "17:00",
                    "priority": 1 if record["type_course"] == "A/R" else 2,
                    "client_name": record["nom_prenom"],
                    "type_course": record["type_course"]
                }
                bookings.append(booking)

        if len(bookings) < min_courses_per_day:
            continue

        # Créer les drivers avec positions initiales (centre de Genève)
        drivers_data = []
        for i, driver_name in enumerate(drivers):
            # Position initiale aléatoire dans le centre de Genève
            base_lat = 46.2044 + np.random.uniform(-0.05, 0.05)
            base_lng = 6.1432 + np.random.uniform(-0.05, 0.05)

            driver = {
                "id": f"driver_{i}",
                "name": driver_name,
                "lat": base_lat,
                "lng": base_lng,
                "available": True,
                "current_load": 0,
                "max_load": 10
            }
            drivers_data.append(driver)

        # Calculer les distances et temps
        total_distance = 0
        total_time = 0

        for booking in bookings:
            # Distance pickup -> dropoff
            dist = haversine_distance(
                booking["pickup_lat"], booking["pickup_lng"],
                booking["dropoff_lat"], booking["dropoff_lng"]
            )
            total_distance += dist

            # Temps estimé (vitesse moyenne 30 km/h en ville)
            time_minutes = (dist / 30) * 60
            total_time += time_minutes

        # Créer le dispatch RL
        dispatch = {
            "date": date_str,
            "num_bookings": len(bookings),
            "num_drivers": len(drivers),
            "bookings": bookings,
            "drivers": drivers_data,
            "total_distance_km": round(total_distance, 2),
            "total_time_minutes": round(total_time, 2),
            "avg_distance_per_booking": round(total_distance / len(bookings), 2),
            "avg_time_per_booking": round(total_time / len(bookings), 2),
            "driver_names": drivers,
            "course_types": {r["type_course"] for r in records if r["type_course"]}
        }

        rl_dispatches.append(dispatch)

    print("✅ {len(rl_dispatches)} dispatches créés")

    # Statistiques
    if rl_dispatches:
        total_bookings = sum(d["num_bookings"] for d in rl_dispatches)
        total_drivers = sum(d["num_drivers"] for d in rl_dispatches)
        total_bookings / len(rl_dispatches)
        total_drivers / len(rl_dispatches)

        print("📊 Statistiques :")
        print("   - Total bookings : {total_bookings}")
        print("   - Total drivers : {total_drivers}")
        print("   - Moyenne bookings/dispatch : {avg_bookings")
        print("   - Moyenne drivers/dispatch : {avg_drivers")

    # Sauvegarder
    output_data = {
        "dispatches": rl_dispatches,
        "metadata": {
            "source_file": cleaned_data_file,
            "conversion_date": datetime.now(tz=UTC).isoformat(),
            "total_dispatches": len(rl_dispatches),
            "total_bookings": sum(d["num_bookings"] for d in rl_dispatches),
            "total_drivers": sum(d["num_drivers"] for d in rl_dispatches),
            "date_range": {
                "start": min(d["date"] for d in rl_dispatches) if rl_dispatches else None,
                "end": max(d["date"] for d in rl_dispatches) if rl_dispatches else None
            }
        }
    }

    with Path(output_file, "w", encoding="utf-8").open() as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print("💾 Données sauvegardées : {output_file}")
    print()
    print("🚀 Prochaine étape : Lancer l'entraînement RL")
    print("   docker-compose exec api python scripts/rl_train_offline.py --cleaned")


def train_with_cleaned_data():
    """Lance l'entraînement RL avec les données nettoyées."""
    print("=" * 80)
    print("🧠 ENTRAÎNEMENT RL AVEC DONNÉES NETTOYÉES")
    print("=" * 80)

    # Convertir d'abord les données
    convert_cleaned_data_to_rl()

    # Importer et lancer l'entraînement
    from scripts.rl_train_offline import train_offline

    print("\n🚀 Lancement de l'entraînement RL...")
    train_offline(
        historical_data_file="/app/data/rl/historical_dispatches_cleaned.json",
        num_episodes=0.15000,  # Plus d'épisodes pour plus de données
        save_path="/app/data/rl/models/dispatch_optimized_v3_cleaned.pth",
        learning_rate=0.00001,
        batch_size=64,
        target_update_freq=0.100,
    )


if __name__ == "__main__":

    if "--convert-only" in sys.argv:
        convert_cleaned_data_to_rl()
    else:
        train_with_cleaned_data()
