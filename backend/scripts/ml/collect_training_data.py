"""
Script de collecte de données pour l'entraînement du modèle ML de prédiction de retards.

Extrait les données historiques des 90 derniers jours et calcule les features nécessaires.

Usage:
    python scripts/ml/collect_training_data.py [--days 90] [--output data/training_data.csv]
"""
# ruff: noqa: T201, DTZ005, DTZ011
# print(), datetime sans tz sont intentionnels dans les scripts ML

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import and_

# Imports Flask app
from app import create_app
from ext import db
from models import Assignment, Booking, Driver
from shared.geo_utils import haversine_distance


def extract_features_from_assignment(
    assignment: Assignment,
    booking: Booking,
    driver: Driver
) -> dict[str, Any]:
    """
    Extrait les features d'une assignation pour le ML.
    Features extraites :
    - time_of_day : Heure de la journée (0-23)
    - day_of_week : Jour de la semaine (0-6, 0=Lundi)
    - month : Mois de l'année (1-12)
    - distance_km : Distance pickup → dropoff en km
    - duration_seconds : Durée réelle du trajet (si disponible)
    - is_medical : Course médicale (0/1)
    - is_urgent : Course urgente (0/1)
    - is_round_trip : Aller-retour (0/1)
    - driver_total_bookings : Nombre total de courses du chauffeur
    - booking_priority : Priorité estimée (0-1)
    - traffic_density : Densité trafic estimée (0-1)
    - weather_factor : Facteur météo (0.5 neutre)
    - actual_delay_minutes : Retard réel (TARGET pour ML)
    """
    scheduled_time = booking.scheduled_time or datetime.now()

    # Time features
    time_of_day = scheduled_time.hour
    day_of_week = scheduled_time.weekday()
    month = scheduled_time.month

    # Distance
    try:
        pickup_lat_val = getattr(booking, 'pickup_lat', None)
        pickup_lon_val = getattr(booking, 'pickup_lon', None)
        dropoff_lat_val = getattr(booking, 'dropoff_lat', None)
        dropoff_lon_val = getattr(booking, 'dropoff_lon', None)

        pickup_lat = float(pickup_lat_val) if pickup_lat_val is not None else 0.0
        pickup_lon = float(pickup_lon_val) if pickup_lon_val is not None else 0.0
        dropoff_lat = float(dropoff_lat_val) if dropoff_lat_val is not None else 0.0
        dropoff_lon = float(dropoff_lon_val) if dropoff_lon_val is not None else 0.0

        if all([pickup_lat, pickup_lon, dropoff_lat, dropoff_lon]):
            distance_km = haversine_distance(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon)
        else:
            distance_meters_val = getattr(booking, 'distance_meters', None)
            distance_km = distance_meters_val / 1000.0 if distance_meters_val else 0.0
    except Exception:
        distance_km = 0.0

    # Duration
    duration_seconds_val = getattr(booking, 'duration_seconds', None)
    duration_seconds = duration_seconds_val if duration_seconds_val else 0

    # Booking characteristics
    medical_facility_val = getattr(booking, 'medical_facility', None)
    is_medical = 1.0 if medical_facility_val else 0.0
    is_urgent_val = getattr(booking, 'is_urgent', False)
    is_urgent = 1.0 if is_urgent_val else 0.0
    is_round_trip_val = getattr(booking, 'is_round_trip', False)
    is_round_trip = 1.0 if is_round_trip_val else 0.0

    # Driver features (nombre total de bookings comme proxy de l'expérience)
    driver_total_bookings = len(getattr(driver, 'assignments', [])) if hasattr(driver, 'assignments') else 0

    # Priority
    booking_priority = 0.8 if (is_medical or is_urgent) else 0.5

    # Traffic density (estimation basée sur l'heure)
    # Heures de pointe : 7-9h et 17-19h = haute densité
    if time_of_day in [7, 8, 17, 18]:
        traffic_density = 0.8
    elif time_of_day in [6, 9, 16, 19]:
        traffic_density = 0.6
    else:
        traffic_density = 0.3

    # Weather (neutre par défaut - pourrait être enrichi avec API météo)
    weather_factor = 0.5

    # TARGET: Calculer le retard réel
    # Retard = (actual_pickup_at - planned_pickup_at) en minutes
    actual_delay_minutes = 0.0

    planned_pickup_val = getattr(assignment, 'planned_pickup_at', None)
    actual_pickup_val = getattr(assignment, 'actual_pickup_at', None)

    if planned_pickup_val is not None and actual_pickup_val is not None:
        try:
            delay_seconds = (actual_pickup_val - planned_pickup_val).total_seconds()
            actual_delay_minutes = delay_seconds / 60.0
        except Exception:
            actual_delay_minutes = 0.0
    else:
        delay_seconds_val = getattr(assignment, 'delay_seconds', None)
        if delay_seconds_val:
            # Utiliser le delay_seconds si disponible
            actual_delay_minutes = delay_seconds_val / 60.0

    return {
        # Features temporelles
        "time_of_day": time_of_day,
        "day_of_week": day_of_week,
        "month": month,

        # Features spatiales
        "distance_km": distance_km,
        "duration_seconds": duration_seconds,

        # Features booking
        "is_medical": is_medical,
        "is_urgent": is_urgent,
        "is_round_trip": is_round_trip,
        "booking_priority": booking_priority,

        # Features driver
        "driver_total_bookings": driver_total_bookings,

        # Features contextuelles
        "traffic_density": traffic_density,
        "weather_factor": weather_factor,

        # IDs pour traçabilité
        "booking_id": booking.id,
        "driver_id": driver.id,
        "assignment_id": assignment.id,
        "company_id": booking.company_id,

        # TARGET (variable à prédire)
        "actual_delay_minutes": actual_delay_minutes,
    }


def collect_training_data(days: int = 90, company_id: int | None = None) -> pd.DataFrame:
    """
    Collecte les données d'entraînement des N derniers jours.

    Args:
        days: Nombre de jours à extraire (défaut: 90)
        company_id: ID de la company (None = toutes)

    Returns:
        DataFrame avec les features et labels
    """
    print(f"\n{'='*70}")
    print("COLLECTE DE DONNÉES ML - DÉMARRAGE")
    print(f"{'='*70}")
    print(f"Période : {days} derniers jours")
    print(f"Company ID : {company_id or 'Toutes'}")
    print(f"{'='*70}\n")

    # Calculer la période
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    print(f"📅 Extraction du {start_date} au {end_date}")

    # Query pour récupérer les assignments avec bookings et drivers
    query = (
        db.session.query(Assignment, Booking, Driver)
        .join(Booking, Assignment.booking_id == Booking.id)
        .join(Driver, Assignment.driver_id == Driver.id)
        .filter(
            and_(
                Booking.scheduled_time >= start_date,
                Booking.scheduled_time <= end_date,
                Assignment.actual_pickup_at.isnot(None),  # Seulement les courses complétées
            )
        )
    )

    if company_id:
        query = query.filter(Booking.company_id == company_id)

    # Eager loading pour éviter N+1 queries
    results = query.all()

    print(f"✅ {len(results)} assignments trouvées avec pickup réel")

    if len(results) == 0:
        print("⚠️ ATTENTION: Aucune donnée trouvée!")
        print("   - Vérifiez que vous avez des assignments avec actual_pickup_at")
        print("   - Essayez d'augmenter la période (--days 180)")
        return pd.DataFrame()

    # Extraire les features
    print("\n🔧 Extraction des features...")
    data_records = []

    for i, (assignment, booking, driver) in enumerate(results, 1):
        if i % 100 == 0:
            print(f"   Traité {i}/{len(results)} assignments...")

        try:
            features = extract_features_from_assignment(assignment, booking, driver)
            data_records.append(features)
        except Exception as e:
            print(f"⚠️ Erreur assignment {assignment.id}: {e}")
            continue

    print(f"✅ {len(data_records)} enregistrements créés")

    # Créer DataFrame
    df = pd.DataFrame(data_records)

    # Statistiques rapides
    print(f"\n{'='*70}")
    print("STATISTIQUES DU DATASET")
    print(f"{'='*70}")
    print(f"Taille : {len(df)} lignes x {len(df.columns)} colonnes")
    print(f"\nColonnes : {', '.join(df.columns)}")
    print(f"\nRetard moyen : {df['actual_delay_minutes'].mean():.2f} minutes")
    print(f"Retard médian : {df['actual_delay_minutes'].median():.2f} minutes")
    print(f"Retard max : {df['actual_delay_minutes'].max():.2f} minutes")
    print(f"Retard min : {df['actual_delay_minutes'].min():.2f} minutes")
    print(f"\n% courses avec retard (>5min) : {(df['actual_delay_minutes'] > 5).sum() / len(df) * 100:.1f}%")
    print(f"% courses en avance (<0min) : {(df['actual_delay_minutes'] < 0).sum() / len(df) * 100:.1f}%")
    print(f"{'='*70}\n")

    return df


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Collecte de données pour entraînement ML")
    parser.add_argument("--days", type=int, default=90, help="Nombre de jours à extraire (défaut: 90)")
    parser.add_argument("--company-id", type=int, default=None, help="ID de la company (défaut: toutes)")
    parser.add_argument("--output", type=str, default="data/ml/training_data.csv", help="Fichier de sortie CSV")

    args = parser.parse_args()

    # Créer l'app Flask pour accéder à la DB
    app = create_app()

    with app.app_context():
        # Collecter les données
        df = collect_training_data(days=args.days, company_id=args.company_id)

        if df.empty:
            print("❌ ERREUR: Dataset vide. Abandon.")
            sys.exit(1)

        # Créer le dossier de sortie si nécessaire
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarder en CSV
        df.to_csv(output_path, index=False)
        print(f"✅ CSV sauvegardé : {output_path}")

        # Sauvegarder aussi en JSON pour flexibilité
        json_path = output_path.with_suffix('.json')
        df.to_json(json_path, orient='records', indent=2)
        print(f"✅ JSON sauvegardé : {json_path}")

        # Créer un fichier de métadonnées
        metadata = {
            "created_at": datetime.now().isoformat(),
            "days_extracted": args.days,
            "company_id": args.company_id,
            "total_records": len(df),
            "start_date": df['booking_id'].min() if not df.empty else None,
            "end_date": df['booking_id'].max() if not df.empty else None,
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
        print("✅ COLLECTE TERMINÉE AVEC SUCCÈS !")
        print(f"{'='*70}")
        print("\nFichiers créés :")
        print(f"  - {output_path}")
        print(f"  - {json_path}")
        print(f"  - {metadata_path}")
        print("\nProchaine étape : Analyse exploratoire (EDA)")
        print("  → python scripts/ml/analyze_data.py")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

