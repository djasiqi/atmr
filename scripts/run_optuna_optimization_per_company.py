#!/usr/bin/env python3
"""
Script pour créer des études Optuna personnalisées par entreprise
Objectif: Optimiser la planification pour chaque entreprise selon sa configuration
- Entreprise 1: 3 chauffeurs
- Entreprise 2: 4 chauffeurs + 1 urgence
- Entreprise 3: 2 chauffeurs + 1 régulier
etc.
"""

import os
import sys
from pathlib import Path
from urllib.parse import quote_plus

# Ajouter le backend au path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

# Import après modification du path
from services.rl.hyperparameter_tuner import HyperparameterTuner  # noqa: E402

# Configuration PostgreSQL
RL_POSTGRES_USER = os.getenv("RL_POSTGRES_USER", "atmr")
RL_POSTGRES_PASSWORD = os.getenv("RL_POSTGRES_PASSWORD")
RL_POSTGRES_DB = os.getenv("RL_POSTGRES_DB", "atmr_rl_db")
RL_POSTGRES_HOST = os.getenv("RL_POSTGRES_HOST", "rl-postgres")
RL_POSTGRES_PORT = os.getenv("RL_POSTGRES_PORT", "5432")

POSTGRES_USER = os.getenv("POSTGRES_USER", "atmr_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB", "atmr_db")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

if not RL_POSTGRES_PASSWORD:
    print("❌ RL_POSTGRES_PASSWORD non défini")
    sys.exit(1)

if not POSTGRES_PASSWORD:
    print("❌ POSTGRES_PASSWORD non défini - nécessaire pour charger les entreprises")
    sys.exit(1)

# Construire les URLs PostgreSQL
ENCODED_RL_PASSWORD = quote_plus(RL_POSTGRES_PASSWORD)
RL_POSTGRES_URL = (
    f"postgresql://{RL_POSTGRES_USER}:{ENCODED_RL_PASSWORD}"
    f"@{RL_POSTGRES_HOST}:{RL_POSTGRES_PORT}/{RL_POSTGRES_DB}"
)

ENCODED_PASSWORD = quote_plus(POSTGRES_PASSWORD)
POSTGRES_URL = (
    f"postgresql://{POSTGRES_USER}:{ENCODED_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Configuration de l'optimisation
N_TRIALS = int(os.getenv("OPTUNA_N_TRIALS", "30"))  # Moins de trials par entreprise
N_TRAINING_EPISODES = int(os.getenv("OPTUNA_TRAINING_EPISODES", "150"))
N_EVAL_EPISODES = int(os.getenv("OPTUNA_EVAL_EPISODES", "15"))

# Période de données pour l'entraînement
# Options: "day", "week", "month", "custom"
DATA_PERIOD = os.getenv("OPTUNA_DATA_PERIOD", "week")  # Par défaut: semaine
CUSTOM_DAYS = int(os.getenv("OPTUNA_CUSTOM_DAYS", "7"))  # Si custom, nombre de jours

# Filtrer par entreprise spécifique (optionnel)
COMPANY_ID = os.getenv(
    "OPTUNA_COMPANY_ID"
)  # Si défini, optimise seulement cette entreprise


def load_companies_configuration(postgres_url: str):
    """Charge la configuration de chaque entreprise (chauffeurs, types)"""
    try:
        from sqlalchemy import create_engine, text
        import pandas as pd

        print("📦 Chargement des configurations des entreprises...")

        engine = create_engine(postgres_url)
        with engine.connect() as conn:
            # Charger les entreprises avec leurs chauffeurs
            query = text("""
                SELECT 
                    c.id as company_id,
                    c.name as company_name,
                    COUNT(DISTINCT d.id) as total_drivers,
                    COUNT(DISTINCT CASE WHEN d.driver_type = 'EMERGENCY' THEN d.id END) as emergency_drivers,
                    COUNT(DISTINCT CASE WHEN d.driver_type = 'REGULAR' THEN d.id END) as regular_drivers,
                    COUNT(DISTINCT CASE WHEN d.is_active = true AND d.is_available = true THEN d.id END) as active_drivers
                FROM company c
                LEFT JOIN driver d ON c.id = d.company_id
                WHERE c.id IS NOT NULL
                GROUP BY c.id, c.name
                HAVING COUNT(DISTINCT d.id) > 0
                ORDER BY c.id
            """)

            df = pd.read_sql(query, conn)

        if df.empty:
            print("⚠️  Aucune entreprise avec chauffeurs trouvée")
            return []

        companies = []
        for _, row in df.iterrows():
            company = {
                "id": int(row["company_id"]),
                "name": str(row["company_name"]),
                "total_drivers": int(row["total_drivers"]),
                "emergency_drivers": int(row["emergency_drivers"]),
                "regular_drivers": int(row["regular_drivers"]),
                "active_drivers": int(row["active_drivers"]),
            }
            companies.append(company)

        print(f"✅ {len(companies)} entreprises trouvées")
        for company in companies:
            print(
                f"   - {company['name']} (ID: {company['id']}): "
                f"{company['regular_drivers']} réguliers, "
                f"{company['emergency_drivers']} urgence, "
                f"{company['total_drivers']} total"
            )

        return companies

    except Exception as e:
        print(f"❌ Erreur lors du chargement des entreprises: {e}")
        import traceback

        traceback.print_exc()
        return []


def load_company_bookings(
    postgres_url: str,
    company_id: int,
    period: str = "week",
    custom_days: int = 7,
    limit: int = 500,
):
    """Charge les bookings réels d'une entreprise sur une période donnée

    Args:
        postgres_url: URL de connexion PostgreSQL
        company_id: ID de l'entreprise
        period: Période de données ("day", "week", "month", "custom")
        custom_days: Nombre de jours si period="custom"
        limit: Nombre maximum de bookings à charger
    """
    try:
        from sqlalchemy import create_engine, text
        import pandas as pd

        # Déterminer la période de dates
        if period == "day":
            date_start = "CURRENT_DATE"
            date_end = "CURRENT_DATE + INTERVAL '1 day'"
        elif period == "week":
            date_start = "CURRENT_DATE - INTERVAL '7 days'"
            date_end = "CURRENT_DATE + INTERVAL '1 day'"
        elif period == "month":
            date_start = "CURRENT_DATE - INTERVAL '30 days'"
            date_end = "CURRENT_DATE + INTERVAL '1 day'"
        elif period == "custom":
            date_start = f"CURRENT_DATE - INTERVAL '{custom_days} days'"
            date_end = "CURRENT_DATE + INTERVAL '1 day'"
        else:
            # Par défaut: semaine
            date_start = "CURRENT_DATE - INTERVAL '7 days'"
            date_end = "CURRENT_DATE + INTERVAL '1 day'"

        engine = create_engine(postgres_url)
        with engine.connect() as conn:
            query = text(f"""
                SELECT 
                    b.id,
                    b.pickup_lat,
                    b.pickup_lon,
                    b.dropoff_lat,
                    b.dropoff_lon,
                    b.scheduled_time,
                    b.is_urgent,
                    b.company_id,
                    b.status,
                    b.completed_at
                FROM booking b
                WHERE b.company_id = :company_id
                    AND b.scheduled_time >= {date_start}
                    AND b.scheduled_time < {date_end}
                    AND b.status IN ('pending', 'confirmed', 'accepted', 'assigned', 'completed')
                    AND b.pickup_lat IS NOT NULL
                    AND b.pickup_lon IS NOT NULL
                ORDER BY b.scheduled_time DESC
                LIMIT :limit
            """)

            df = pd.read_sql(
                query, conn, params={"company_id": company_id, "limit": limit}
            )

        if df.empty:
            return None

        # Calculer les statistiques
        stats = {
            "total_bookings": len(df),
            "urgent_bookings": int(df["is_urgent"].sum())
            if "is_urgent" in df.columns
            else 0,
            "avg_lat": float(df["pickup_lat"].mean()),
            "avg_lon": float(df["pickup_lon"].mean()),
            "time_range": (
                df["scheduled_time"].min(),
                df["scheduled_time"].max(),
            )
            if "scheduled_time" in df.columns
            else None,
        }

        return stats

    except Exception as e:
        print(f"⚠️  Erreur lors du chargement des bookings: {e}")
        return None


class CompanySpecificHyperparameterTuner(HyperparameterTuner):
    """Tuner personnalisé pour une entreprise spécifique"""

    def __init__(self, company_config: dict, *args, **kwargs):
        self.company_config = company_config
        # Ajuster num_drivers selon la configuration de l'entreprise
        self.company_num_drivers = (
            company_config["active_drivers"] or company_config["total_drivers"]
        )
        super().__init__(*args, **kwargs)

    def _suggest_hyperparameters(self, trial):
        """Suggère des hyperparamètres adaptés à la configuration de l'entreprise"""
        config = super()._suggest_hyperparameters(trial)

        # Ajuster num_drivers selon l'entreprise
        # Utiliser le nombre réel de chauffeurs actifs de l'entreprise
        config["num_drivers"] = max(
            3, min(self.company_num_drivers, 20)
        )  # Entre 3 et 20

        # Ajuster max_bookings selon le nombre de bookings typiques de l'entreprise
        # Si l'entreprise a peu de chauffeurs, elle a probablement moins de bookings
        if self.company_num_drivers <= 3:
            config["max_bookings"] = trial.suggest_int("max_bookings", 5, 15)
        elif self.company_num_drivers <= 5:
            config["max_bookings"] = trial.suggest_int("max_bookings", 10, 25)
        else:
            config["max_bookings"] = trial.suggest_int("max_bookings", 15, 50)

        return config


def optimize_company(company: dict, postgres_url: str, rl_postgres_url: str):
    """Optimise les hyperparamètres pour une entreprise spécifique"""
    company_id = company["id"]
    company_name = company["name"]

    print()
    print("=" * 60)
    print(f"🏢 OPTIMISATION POUR: {company_name} (ID: {company_id})")
    print("=" * 60)
    print(
        f"   Chauffeurs: {company['regular_drivers']} réguliers, {company['emergency_drivers']} urgence"
    )
    print(f"   Total actifs: {company['active_drivers']}")

    # Charger les statistiques des bookings de l'entreprise
    bookings_stats = load_company_bookings(
        postgres_url, company_id, period=DATA_PERIOD, custom_days=CUSTOM_DAYS
    )
    if bookings_stats:
        print(f"   📊 Période de données: {bookings_stats['period']}")
        print(f"   📦 Bookings chargés: {bookings_stats['total_bookings']}")
        print(f"   🚨 Bookings urgents: {bookings_stats['urgent_bookings']}")
        print(
            f"   ✅ Bookings complétés: {bookings_stats.get('completed_bookings', 0)}"
        )
        print(f"   ⏳ Bookings en attente: {bookings_stats.get('pending_bookings', 0)}")
        if bookings_stats["time_range"]:
            print(
                f"   📅 Période: {bookings_stats['time_range'][0]} à {bookings_stats['time_range'][1]}"
            )
        print(
            f"   📍 Centre géographique: ({bookings_stats['avg_lat']:.4f}, {bookings_stats['avg_lon']:.4f})"
        )
    else:
        print("   ⚠️  Aucun booking trouvé pour cette période")

    # Créer le nom de l'étude spécifique à l'entreprise
    study_name = f"dqn_optimization_company_{company_id}"

    # Créer le tuner personnalisé
    tuner = CompanySpecificHyperparameterTuner(
        company_config=company,
        n_trials=N_TRIALS,
        n_training_episodes=N_TRAINING_EPISODES,
        n_eval_episodes=N_EVAL_EPISODES,
        study_name=study_name,
        storage=rl_postgres_url,
    )

    try:
        # Lancer l'optimisation
        study = tuner.optimize()

        # Sauvegarder les meilleurs paramètres
        output_path = Path(f"data/rl/optimal_config_company_{company_id}.json")
        tuner.save_best_params(study, output_path=str(output_path))

        print()
        print(f"✅ OPTIMISATION TERMINÉE pour {company_name} !")
        print(f"📊 Meilleur reward: {study.best_value:.2f}")
        print(f"🎯 Meilleur trial: #{study.best_trial.number}")
        print(f"📁 Paramètres sauvegardés: {output_path}")
        print()
        print("🔍 Meilleurs paramètres:")
        for param, value in study.best_params.items():
            print(f"   {param}: {value}")
        print()
        print("🌐 Voir dans le dashboard: https://optuna.lirie.ch")
        print(f"   Recherchez: {study_name}")

        return study

    except Exception as e:
        print(f"❌ Erreur lors de l'optimisation pour {company_name}: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """Fonction principale"""
    print("🚀 Optimisation Optuna personnalisée par entreprise")
    print("=" * 60)
    print(f"📊 Base de données RL: {RL_POSTGRES_DB}")
    print(f"📦 Base de données principale: {POSTGRES_DB}")
    print(f"🔢 Nombre de trials par entreprise: {N_TRIALS}")
    print(f"🎮 Épisodes d'entraînement: {N_TRAINING_EPISODES}")
    print(f"✅ Épisodes d'évaluation: {N_EVAL_EPISODES}")
    print(f"📅 Période de données: {DATA_PERIOD}")
    if DATA_PERIOD == "custom":
        print(f"   Nombre de jours: {CUSTOM_DAYS}")
    if COMPANY_ID:
        print(f"🎯 Entreprise ciblée: {COMPANY_ID}")
    print("=" * 60)
    print()

    # Charger les entreprises
    companies = load_companies_configuration(POSTGRES_URL)

    if not companies:
        print("❌ Aucune entreprise trouvée")
        sys.exit(1)

    # Filtrer par entreprise spécifique si demandé
    if COMPANY_ID:
        companies = [c for c in companies if c["id"] == int(COMPANY_ID)]
        if not companies:
            print(f"❌ Entreprise {COMPANY_ID} non trouvée")
            sys.exit(1)

    # Optimiser chaque entreprise
    results = []
    for company in companies:
        study = optimize_company(company, POSTGRES_URL, RL_POSTGRES_URL)
        if study:
            results.append({"company": company, "study": study})

    # Résumé final
    print()
    print("=" * 60)
    print("📊 RÉSUMÉ DES OPTIMISATIONS")
    print("=" * 60)
    for result in results:
        company = result["company"]
        study = result["study"]
        print(
            f"🏢 {company['name']} (ID: {company['id']}): "
            f"Reward = {study.best_value:.2f} "
            f"(Trial #{study.best_trial.number})"
        )
    print()
    print("🌐 Accédez au dashboard: https://optuna.lirie.ch")
    print("   Vous verrez une étude séparée pour chaque entreprise")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Optimisation interrompue par l'utilisateur")
        print("💡 Les résultats partiels sont sauvegardés dans PostgreSQL")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
