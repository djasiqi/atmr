#!/usr/bin/env python3
"""Routes API pour l'environnement Reinforcement Learning.

Ce module expose des endpoints REST pour interagir avec l'environnement RL,
notamment pour l'optimisation Optuna des hyperparamètres.
"""

import logging
import threading
from typing import Any

from flask import current_app  # pyright: ignore[reportMissingImports]
from flask_restx import (  # pyright: ignore[reportMissingImports]
    Namespace,
    Resource,
    fields,
)

from tasks.rl_tasks import (
    optuna_optimize_impl,
    train_model_with_optimal_params_impl,
)

logger = logging.getLogger(__name__)

# Créer le namespace RL
rl_ns = Namespace("rl", description="Reinforcement Learning API")

# Modèle pour l'optimisation Optuna
optuna_optimize_model = rl_ns.model(
    "OptunaOptimize",
    {
        "company_id": fields.Integer(
            required=False, description="ID de l'entreprise (optionnel)"
        ),
        "data_period": fields.String(
            required=False,
            default="week",
            description="Période des données (week, month, custom)",
        ),
        "n_trials": fields.Integer(
            required=False, default=30, description="Nombre de trials Optuna"
        ),
        "training_episodes": fields.Integer(
            required=False,
            default=150,
            description="Nombre d'épisodes d'entraînement par trial",
        ),
        "eval_episodes": fields.Integer(
            required=False,
            default=15,
            description="Nombre d'épisodes d'évaluation par trial",
        ),
        "custom_days": fields.Integer(
            required=False,
            default=7,
            description="Nombre de jours pour la période 'custom'",
        ),
    },
)


@rl_ns.route("/optuna/optimize")
class RLOptunaOptimize(Resource):
    """Endpoint pour lancer l'optimisation Optuna des hyperparamètres DQN."""

    @rl_ns.expect(optuna_optimize_model, validate=False)
    def post(self):
        """
        Déclenche l'optimisation Optuna pour les hyperparamètres DQN.

        Cette route exécute l'optimisation en arrière-plan dans un thread.
        L'optimisation peut prendre plusieurs heures selon le nombre de trials.

        Retourne immédiatement avec un statut de démarrage (202 Accepted).
        """
        try:
            data = rl_ns.payload or {}
            company_id = data.get("company_id")
            data_period = data.get("data_period", "week")
            n_trials = data.get("n_trials", 30)
            training_episodes = data.get("training_episodes", 150)
            eval_episodes = data.get("eval_episodes", 15)
            custom_days = data.get("custom_days", 7)

            logger.info(
                (
                    "🚀 [RL] Démarrage optimisation Optuna: "
                    "company_id=%s, period=%s, trials=%s"
                ),
                company_id,
                data_period,
                n_trials,
            )

            # Capturer l'application Flask AVANT de créer le thread
            # current_app est un proxy local qui n'est pas disponible dans les threads
            flask_app = current_app._get_current_object()

            # Lancer l'optimisation en arrière-plan (threading)
            def run_optuna_optimization():
                """Fonction exécutée en arrière-plan pour l'optimisation Optuna."""
                try:
                    # Créer un contexte d'application Flask pour le thread
                    # en utilisant l'application Flask capturée avant le thread
                    with flask_app.app_context():
                        result = optuna_optimize_impl(
                            company_id=company_id,
                            data_period=data_period,
                            n_trials=n_trials,
                            training_episodes=training_episodes,
                            eval_episodes=eval_episodes,
                            custom_days=custom_days
                            if data_period == "custom"
                            else None,
                        )

                        logger.info(
                            "✅ [RL] Optimisation Optuna terminée: %s",
                            result.get("status", "unknown"),
                        )

                except Exception:
                    logger.exception(
                        "❌ [RL] Erreur lors de l'exécution de l'optimisation Optuna"
                    )

            # Démarrer le thread en arrière-plan
            optuna_thread = threading.Thread(
                target=run_optuna_optimization,
                daemon=True,
                name="rl-optuna-optimization",
            )
            optuna_thread.start()

            logger.info(
                (
                    "✅ [RL] Thread Optuna démarré (thread_id=%s), "
                    "company_id=%s, trials=%s"
                ),
                optuna_thread.ident,
                company_id,
                n_trials,
            )

            # Construire le nom de l'étude pour l'URL Optuna Dashboard
            if company_id:
                study_name = f"dqn_optimization_company_{company_id}"
            else:
                study_name = "dqn_optimization_all_companies"

            response_data: dict[str, Any] = {
                "message": "Optimisation Optuna démarrée",
                "status": "started",
                "thread_id": str(optuna_thread.ident),
                "config": {
                    "company_id": company_id,
                    "data_period": data_period,
                    "n_trials": n_trials,
                    "training_episodes": training_episodes,
                    "eval_episodes": eval_episodes,
                    "custom_days": custom_days if data_period == "custom" else None,
                },
                "study_name": study_name,
                "note": (
                    "L'optimisation s'exécute en arrière-plan. "
                    f"Consultez https://optuna.lirie.ch pour suivre la progression. "
                    f"Recherchez l'étude '{study_name}' dans le dashboard."
                ),
            }

            return response_data, 202  # 202 Accepted (traitement asynchrone)

        except Exception as e:
            logger.exception("❌ [RL] ERREUR optuna_optimize: %s", e)
            return {"message": "Erreur lors du démarrage de l'optimisation"}, 500


# Modèle pour l'entraînement avec hyperparamètres optimaux
train_optimal_model = rl_ns.model(
    "TrainOptimalModel",
    {
        "config_path": fields.String(
            required=False,
            description="Chemin vers optimal_config.json (optionnel)",
        ),
        "study_name": fields.String(
            required=False,
            description="Nom de l'étude Optuna (optionnel, si config_path non fourni)",
        ),
        "model_output_path": fields.String(
            required=False,
            default="data/rl/models/dqn_optimized.pth",
            description="Chemin de sortie pour le modèle entraîné",
        ),
        "training_episodes": fields.Integer(
            required=False,
            default=1000,
            description="Nombre d'épisodes d'entraînement complet",
        ),
        "eval_episodes": fields.Integer(
            required=False,
            default=50,
            description="Nombre d'épisodes d'évaluation finale",
        ),
        "company_id": fields.Integer(
            required=False, description="ID de l'entreprise (optionnel)"
        ),
    },
)


@rl_ns.route("/train/optimal")
class RLTrainOptimal(Resource):
    """Endpoint pour entraîner un modèle DQN avec les hyperparamètres optimaux."""

    @rl_ns.expect(train_optimal_model, validate=False)
    def post(self):
        """
        Entraîne un modèle DQN complet avec les hyperparamètres optimaux.

        Les hyperparamètres peuvent être chargés depuis:
        - Un fichier optimal_config.json (config_path)
        - Une étude Optuna (study_name)
        - Les hyperparamètres par défaut (si aucun des deux n'est fourni)

        Cette route exécute l'entraînement en arrière-plan dans un thread.
        L'entraînement peut prendre plusieurs heures selon le nombre d'épisodes.

        Retourne immédiatement avec un statut de démarrage (202 Accepted).
        """
        try:
            data = rl_ns.payload or {}
            config_path = data.get("config_path")
            study_name = data.get("study_name")
            model_output_path = data.get(
                "model_output_path", "data/rl/models/dqn_optimized.pth"
            )
            training_episodes = data.get("training_episodes", 1000)
            eval_episodes = data.get("eval_episodes", 50)
            company_id = data.get("company_id")

            logger.info(
                (
                    "🎓 [RL] Démarrage entraînement modèle optimal: "
                    "config_path=%s, study_name=%s, episodes=%s"
                ),
                config_path,
                study_name,
                training_episodes,
            )

            # Capturer l'application Flask AVANT de créer le thread
            flask_app = current_app._get_current_object()

            # Lancer l'entraînement en arrière-plan (threading)
            def run_training():
                """Fonction exécutée en arrière-plan pour l'entraînement."""
                try:
                    with flask_app.app_context():
                        result = train_model_with_optimal_params_impl(
                            config_path=config_path,
                            study_name=study_name,
                            model_output_path=model_output_path,
                            training_episodes=training_episodes,
                            eval_episodes=eval_episodes,
                            company_id=company_id,
                        )

                        logger.info(
                            "✅ [RL] Entraînement terminé: %s",
                            result.get("status", "unknown"),
                        )

                except Exception:
                    logger.exception(
                        "❌ [RL] Erreur lors de l'exécution de l'entraînement"
                    )

            # Démarrer le thread en arrière-plan
            training_thread = threading.Thread(
                target=run_training,
                daemon=True,
                name="rl-model-training",
            )
            training_thread.start()

            logger.info(
                ("✅ [RL] Thread entraînement démarré (thread_id=%s), episodes=%s"),
                training_thread.ident,
                training_episodes,
            )

            response_data: dict[str, Any] = {
                "message": "Entraînement du modèle démarré",
                "status": "started",
                "thread_id": str(training_thread.ident),
                "config": {
                    "config_path": config_path,
                    "study_name": study_name,
                    "model_output_path": model_output_path,
                    "training_episodes": training_episodes,
                    "eval_episodes": eval_episodes,
                    "company_id": company_id,
                },
                "note": (
                    "L'entraînement s'exécute en arrière-plan. "
                    f"Le modèle sera sauvegardé dans {model_output_path} une fois terminé."
                ),
            }

            return response_data, 202  # 202 Accepted (traitement asynchrone)

        except Exception as e:
            logger.exception("❌ [RL] ERREUR train_optimal: %s", e)
            return {"message": "Erreur lors du démarrage de l'entraînement"}, 500
