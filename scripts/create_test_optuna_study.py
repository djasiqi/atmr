#!/usr/bin/env python3
"""
Script pour créer une étude Optuna de test dans PostgreSQL
Utile pour vérifier que Optuna Dashboard fonctionne correctement
"""

import os
import sys
from urllib.parse import quote_plus

import optuna

# Lire les variables d'environnement
RL_POSTGRES_USER = os.getenv("RL_POSTGRES_USER", "atmr_rl_user")
RL_POSTGRES_PASSWORD = os.getenv("RL_POSTGRES_PASSWORD", "atmr_rl_password")
RL_POSTGRES_DB = os.getenv("RL_POSTGRES_DB", "atmr_rl_db")
RL_POSTGRES_HOST = os.getenv("RL_POSTGRES_HOST", "rl-postgres")
RL_POSTGRES_PORT = os.getenv("RL_POSTGRES_PORT", "5432")

# Encoder le mot de passe pour l'URL
ENCODED_PASSWORD = quote_plus(RL_POSTGRES_PASSWORD)

# Construire l'URL PostgreSQL
POSTGRES_URL = f"postgresql://{RL_POSTGRES_USER}:{ENCODED_PASSWORD}@{RL_POSTGRES_HOST}:{RL_POSTGRES_PORT}/{RL_POSTGRES_DB}"

print(
    f"🔗 Connexion à PostgreSQL: {RL_POSTGRES_HOST}:{RL_POSTGRES_PORT}/{RL_POSTGRES_DB}"
)
print("📊 Création d'une étude Optuna de test...")

try:
    # Créer ou charger une étude de test
    study = optuna.create_study(
        study_name="test_study",
        direction="maximize",
        storage=POSTGRES_URL,
        load_if_exists=True,
    )

    print(f"✅ Étude créée/chargée: {study.study_name}")
    print(f"   Study ID: {study._study_id}")

    # Créer quelques trials de test
    print("\n🧪 Création de 5 trials de test...")

    def simple_objective(trial):
        """Fonction objective simple pour les tests"""
        x = trial.suggest_float("x", -10.0, 10.0)
        y = trial.suggest_float("y", -10.0, 10.0)
        # Fonction simple : maximiser -(x^2 + y^2) (maximum à x=0, y=0)
        return -(x**2 + y**2)

    study.optimize(simple_objective, n_trials=5, show_progress_bar=True)

    print("\n✅ Étude de test créée avec succès !")
    print(f"   Nombre de trials: {len(study.trials)}")
    print(f"   Meilleur trial: {study.best_trial.number}")
    print(f"   Meilleure valeur: {study.best_value:.4f}")
    print("\n🌐 Accédez au dashboard: https://optuna.lirie.ch")
    print("   Vous devriez maintenant voir l'étude 'test_study' dans le dashboard")

except Exception as e:
    print(f"❌ Erreur: {e}")
    sys.exit(1)
