# pyright: reportReturnType=false
"""Script de collecte de données historiques pour l'entraînement RL.

Collecte les assignments des X derniers jours pour:
- Analyser les patterns de dispatch
- Créer une baseline heuristique
- Entraîner l'agent RL

Usage:
    python scripts/rl/collect_historical_data.py --days 90
"""
import argparse
import pickle

# Imports relatifs au projet
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import and_

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app import create_app
from models import Assignment


def collect_dispatch_data(days_back: int = 90) -> pd.DataFrame:
    """Collecte les données de dispatch historiques.

    Args:
        days_back: Nombre de jours à collecter

    Returns:
        DataFrame avec les données d'assignment

    """
    print("📊 Collecte des données des {days_back} derniers jours...")

    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days_back)

    # Récupérer tous les assignments avec jointures
    assignments = (
        Assignment.query
        .filter(
            and_(
                Assignment.created_at >= start_date,
                Assignment.created_at <= end_date
            )
        )
        .all()
    )

    print("✅ {len(assignments)} assignments trouvés")

    data = []
    for assign in assignments:
        try:
            booking = assign.booking
            driver = assign.driver

            if not booking or not driver:
                continue

            data.append({
                "assignment_id": assign.id,
                "booking_id": booking.id,
                "driver_id": driver.id,
                "company_id": booking.company_id if hasattr(booking, "company_id") else None,

                # Positions
                "pickup_lat": booking.pickup_lat if hasattr(booking, "pickup_lat") else None,
                "pickup_lon": booking.pickup_lon if hasattr(booking, "pickup_lon") else None,
                "dropoff_lat": booking.dropoff_lat if hasattr(booking, "dropoff_lat") else None,
                "dropoff_lon": booking.dropoff_lon if hasattr(booking, "dropoff_lon") else None,
                "driver_lat": driver.latitude if hasattr(driver, "latitude") else None,
                "driver_lon": driver.longitude if hasattr(driver, "longitude") else None,

                # Timing
                "pickup_time": booking.pickup_time if hasattr(booking, "pickup_time") else None,
                "dropoff_time": booking.dropoff_time if hasattr(booking, "dropoff_time") else None,
                "assignment_time": assign.created_at,

                # Métriques
                "distance_km": assign.distance if hasattr(assign, "distance") else 0.0,
                "duration_minutes": assign.duration if hasattr(assign, "duration") else 0,
                "was_late": getattr(assign, "was_late", False),
                "priority": getattr(booking, "priority", 3),

                # Résultat
                "status": booking.status.value if hasattr(booking, "status") else "unknown",
                "customer_rating": getattr(booking, "rating", None),
                "driver_available": driver.is_available if hasattr(driver, "is_available") else True,

                # Contexte
                "hour_of_day": assign.created_at.hour,
                "day_of_week": assign.created_at.weekday(),
            })
        except Exception:
            print("⚠️  Erreur sur assignment {assign.id}: {e}")
            continue

    df = pd.DataFrame(data)

    # Nettoyer les données
    print("\n🧹 Nettoyage des données...")
    len(df)

    # Retirer les lignes avec positions manquantes
    df = df.dropna(subset=["pickup_lat", "pickup_lon", "driver_lat", "driver_lon"])

    # Retirer les distances aberrantes (> 100km)
    df = df[df["distance_km"] < 100]

    # Retirer les durées aberrantes (> 3h)
    df = df[df["duration_minutes"] < 180]

    print("  Lignes initiales: {initial_count}")
    print("  Lignes nettoyées: {len(df)}")
    print("  Lignes retirées: {initial_count - len(df)}")

    return df


def calculate_statistics(df: pd.DataFrame) -> dict:
    """Calcule des statistiques sur les données.

    Args:
        df: DataFrame des assignments

    Returns:
        Dictionnaire de statistiques

    """
    return {
        "total_assignments": len(df),
        "avg_distance_km": df["distance_km"].mean(),
        "avg_duration_min": df["duration_minutes"].mean(),
        "late_rate": df["was_late"].mean() if "was_late" in df.columns else 0.0,
        "avg_rating": df["customer_rating"].mean() if "customer_rating" in df.columns else None,

        # Par heure
        "assignments_by_hour": df.groupby("hour_of_day").size().to_dict(),

        # Par jour de semaine
        "assignments_by_weekday": df.groupby("day_of_week").size().to_dict(),

        # Distances
        "distance_p50": df["distance_km"].median(),
        "distance_p90": df["distance_km"].quantile(0.9),
        "distance_p99": df["distance_km"].quantile(0.99),
    }



def create_baseline_policy() -> dict:
    """Crée une politique baseline heuristique.

    Returns:
        Dictionnaire décrivant la politique

    """
    return {
        "name": "nearest_driver",
        "version": "1.0",
        "description": "Assigne toujours au chauffeur disponible le plus proche",
        "algorithm": "greedy_distance",
        "parameters": {
            "max_distance_km": 20.0,
            "consider_traffic": False,
            "consider_workload": False,
        },
        "expected_performance": {
            "avg_distance_km": 7.5,
            "late_rate": 0.15,
            "completion_rate": 0.85,
        }
    }


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Collecte données historiques pour RL"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Nombre de jours à collecter (défaut: 90)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/rl",
        help="Répertoire de sortie"
    )

    args = parser.parse_args()

    print("="*60)
    print("🚀 COLLECTE DE DONNÉES HISTORIQUES - RL")
    print("="*60)

    # Créer l'application Flask pour accès DB
    app = create_app()

    with app.app_context():
        # Collecter les données
        df = collect_dispatch_data(days_back=args.days)

        if df.empty:
            print("\n❌ Aucune donnée trouvée!")
            return

        # Calculer les statistiques
        print("\n📈 Calcul des statistiques...")
        stats = calculate_statistics(df)

        print("\n📊 STATISTIQUES:")
        print("  Total assignments: {stats['total_assignments']}")
        print("  Distance moyenne: {stats['avg_distance_km']")
        print("  Durée moyenne: {stats['avg_duration_min']")
        print("  Taux de retard: {stats['late_rate']")
        if stats["avg_rating"]:
            print("  Note moyenne: {stats['avg_rating']")

        print("\n  Distance P50: {stats['distance_p50']")
        print("  Distance P90: {stats['distance_p90']")
        print("  Distance P99: {stats['distance_p99']")

        # Créer le répertoire de sortie
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sauvegarder les données
        csv_path = output_dir / "historical_assignments.csv"
        df.to_csv(csv_path, index=False)
        print("\n💾 Données sauvegardées: {csv_path}")

        # Sauvegarder les statistiques
        stats_path = output_dir / "statistics.pkl"
        with Path(stats_path, "wb").open() as f:
            pickle.dump(stats, f)
        print("💾 Statistiques sauvegardées: {stats_path}")

        # Créer et sauvegarder la politique baseline
        baseline = create_baseline_policy()
        baseline_path = output_dir / "baseline_policy.pkl"
        with Path(baseline_path, "wb").open() as f:
            pickle.dump(baseline, f)
        print("💾 Politique baseline sauvegardée: {baseline_path}")

        print("\n✅ Collecte terminée!")
        print("   {len(df)} assignments prêts pour l'entraînement")
        print("="*60)


if __name__ == "__main__":
    main()

