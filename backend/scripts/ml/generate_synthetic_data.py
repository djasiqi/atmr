"""
Génère des données synthétiques réalistes pour l'entraînement du modèle ML.

Usage:
    python scripts/ml/generate_synthetic_data.py --count 1000 --output data/ml/training_data.csv
"""
# ruff: noqa: T201, DTZ005, DTZ011, S311
# print(), datetime, random sont intentionnels pour génération de données

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


def generate_synthetic_training_data(count: int = 1000) -> pd.DataFrame:
    """
    Génère des données synthétiques réalistes pour l'entraînement ML.
    Args:
        count: Nombre d'échantillons à générer
    Returns:
        DataFrame avec features et labels synthétiques
    """
    print(f"\n{'='*70}")
    print("GÉNÉRATION DE DONNÉES SYNTHÉTIQUES - DÉMARRAGE")
    print(f"{'='*70}")
    print(f"Nombre d'échantillons : {count}")
    print(f"{'='*70}\n")

    random.seed(42)  # Reproductibilité

    data_records = []

    for i in range(count):
        if (i + 1) % 100 == 0:
            print(f"Généré {i + 1}/{count} échantillons...")

        # Génération réaliste basée sur patterns Genève

        # Time features (distribution réaliste)
        # Heures de pointe : 7-9h (30%), 12-14h (20%), 17-19h (30%), autres (20%)
        rand = random.random()
        if rand < 0.30:  # Matin
            time_of_day = random.randint(7, 9)
        elif rand < 0.50:  # Midi
            time_of_day = random.randint(12, 14)
        elif rand < 0.80:  # Soir
            time_of_day = random.randint(17, 19)
        else:  # Autres
            time_of_day = random.randint(6, 22)

        day_of_week = random.randint(0, 6)  # 0=Lundi, 6=Dimanche
        month = random.randint(1, 12)

        # Distance (distribution log-normale, moyenne ~8km pour Genève)
        distance_km = max(0.5, random.lognormvariate(1.8, 0.8))  # Moyenne ~7-8km

        # Duration (corrélé à distance, ~5-10 min/km selon trafic)
        base_duration = distance_km * 7 * 60  # ~7 min/km en moyenne
        duration_seconds = int(base_duration * random.uniform(0.7, 1.5))

        # Booking characteristics
        is_medical = random.random() < 0.15  # 15% de courses médicales
        is_urgent = random.random() < 0.10  # 10% de courses urgentes
        is_round_trip = random.random() < 0.25  # 25% aller-retour

        booking_priority = 0.8 if (is_medical or is_urgent) else 0.5

        # Driver features (distribution réaliste)
        # Expérience : 70% drivers expérimentés (>100 courses)
        driver_total_bookings = random.randint(100, 500) if random.random() < 0.70 else random.randint(10, 100)

        # Traffic density (corrélé à l'heure et jour)
        if time_of_day in [7, 8, 17, 18]:  # Heures de pointe
            base_traffic = 0.8
        elif time_of_day in [6, 9, 16, 19]:
            base_traffic = 0.6
        elif time_of_day in [12, 13]:
            base_traffic = 0.5
        else:
            base_traffic = 0.3

        # Weekend = moins de trafic
        if day_of_week >= 5:  # Samedi/Dimanche
            base_traffic *= 0.6

        traffic_density = min(1.0, base_traffic * random.uniform(0.8, 1.2))

        # Weather (distribution réaliste Suisse)
        # 70% beau temps, 20% pluie légère, 10% mauvais temps
        rand = random.random()
        if rand < 0.70:
            weather_factor = random.uniform(0.4, 0.6)  # Beau
        elif rand < 0.90:
            weather_factor = random.uniform(0.6, 0.8)  # Pluie légère
        else:
            weather_factor = random.uniform(0.8, 1.0)  # Mauvais temps

        # TARGET: Calcul du retard réaliste (modèle causal simplifié)
        # Retard = f(distance, trafic, météo, urgence, expérience driver)

        # Baseline delay
        base_delay = 0.0

        # Distance influence (+0.5 min par km au-dessus de 10km)
        if distance_km > 10:
            base_delay += (distance_km - 10) * 0.5

        # Traffic (heures de pointe = +3 à +8 min)
        base_delay += traffic_density * 8.0

        # Weather (mauvais temps = +2 à +10 min)
        base_delay += (weather_factor - 0.5) * 10.0

        # Urgence (paradoxe: plus urgent = plus de pression = plus de retard potentiel)
        if is_urgent:
            base_delay += random.uniform(-2, 3)  # Peut être en avance ou en retard

        # Expérience driver (moins expérimenté = plus de retard)
        if driver_total_bookings < 50:
            base_delay += random.uniform(1, 5)
        elif driver_total_bookings > 200:
            base_delay -= random.uniform(0, 2)  # Drivers expérimentés = moins de retard

        # Ajouter du bruit réaliste (±5 minutes)
        noise = random.gauss(0, 2.5)
        actual_delay_minutes = base_delay + noise

        # Limiter les valeurs extrêmes
        actual_delay_minutes = max(-15.0, min(60.0, actual_delay_minutes))

        # IDs synthétiques
        booking_id = i + 1
        driver_id = random.randint(1, 20)  # 20 drivers simulés
        assignment_id = i + 1
        company_id = 1

        record = {
            # Features temporelles
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "month": month,

            # Features spatiales
            "distance_km": round(distance_km, 2),
            "duration_seconds": duration_seconds,

            # Features booking
            "is_medical": 1.0 if is_medical else 0.0,
            "is_urgent": 1.0 if is_urgent else 0.0,
            "is_round_trip": 1.0 if is_round_trip else 0.0,
            "booking_priority": booking_priority,

            # Features driver
            "driver_total_bookings": driver_total_bookings,

            # Features contextuelles
            "traffic_density": round(traffic_density, 3),
            "weather_factor": round(weather_factor, 3),

            # IDs
            "booking_id": booking_id,
            "driver_id": driver_id,
            "assignment_id": assignment_id,
            "company_id": company_id,

            # TARGET
            "actual_delay_minutes": round(actual_delay_minutes, 2),
        }

        data_records.append(record)

    df = pd.DataFrame(data_records)

    # Statistiques
    print(f"\n{'='*70}")
    print("STATISTIQUES DU DATASET SYNTHÉTIQUE")
    print(f"{'='*70}")
    print(f"Taille : {len(df)} lignes x {len(df.columns)} colonnes")
    print(f"\nRetard moyen : {df['actual_delay_minutes'].mean():.2f} minutes")
    print(f"Retard médian : {df['actual_delay_minutes'].median():.2f} minutes")
    print(f"Écart-type : {df['actual_delay_minutes'].std():.2f} minutes")
    print(f"Retard max : {df['actual_delay_minutes'].max():.2f} minutes")
    print(f"Retard min : {df['actual_delay_minutes'].min():.2f} minutes")
    print(f"\n% courses avec retard (>5min) : {(df['actual_delay_minutes'] > 5).sum() / len(df) * 100:.1f}%")
    print(f"% courses à l'heure (±5min) : {(df['actual_delay_minutes'].abs() <= 5).sum() / len(df) * 100:.1f}%")
    print(f"% courses en avance (<-5min) : {(df['actual_delay_minutes'] < -5).sum() / len(df) * 100:.1f}%")

    # Corrélations intéressantes
    print("\nCorrélations avec retard :")
    correlations = df.corr()['actual_delay_minutes'].sort_values(ascending=False)
    for feature, corr in correlations.items():
        if feature != 'actual_delay_minutes' and abs(corr) > 0.1:
            print(f"  - {feature:25s} : {corr:+.3f}")

    print(f"{'='*70}\n")

    return df


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Génération de données synthétiques pour ML")
    parser.add_argument("--count", type=int, default=5000, help="Nombre d'échantillons (défaut: 5000)")
    parser.add_argument("--output", type=str, default="data/ml/training_data.csv", help="Fichier de sortie CSV")

    args = parser.parse_args()

    # Générer les données
    df = generate_synthetic_training_data(count=args.count)

    if df.empty:
        print("❌ ERREUR: Dataset vide. Abandon.")
        sys.exit(1)

    # Créer le dossier de sortie
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sauvegarder en CSV
    df.to_csv(output_path, index=False)
    print(f"✅ CSV sauvegardé : {output_path}")

    # Sauvegarder en JSON
    json_path = output_path.with_suffix('.json')
    df.to_json(json_path, orient='records', indent=2)
    print(f"✅ JSON sauvegardé : {json_path}")

    # Métadonnées
    metadata = {
        "created_at": datetime.now().isoformat(),
        "type": "synthetic",
        "total_records": len(df),
        "features": list(df.columns),
        "statistics": {
            "mean_delay": float(df['actual_delay_minutes'].mean()),
            "median_delay": float(df['actual_delay_minutes'].median()),
            "std_delay": float(df['actual_delay_minutes'].std()),
            "min_delay": float(df['actual_delay_minutes'].min()),
            "max_delay": float(df['actual_delay_minutes'].max()),
            "pct_delayed": float((df['actual_delay_minutes'] > 5).sum() / len(df) * 100),
        }
    }

    metadata_path = output_path.parent / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Métadonnées sauvegardées : {metadata_path}")

    print(f"\n{'='*70}")
    print("✅ GÉNÉRATION TERMINÉE AVEC SUCCÈS !")
    print(f"{'='*70}")
    print("\nFichiers créés :")
    print(f"  - {output_path}")
    print(f"  - {json_path}")
    print(f"  - {metadata_path}")
    print("\nProchaine étape : Analyse exploratoire (EDA)")
    print(f"  → python scripts/ml/analyze_data.py {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

