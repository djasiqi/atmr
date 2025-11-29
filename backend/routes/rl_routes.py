#!/usr/bin/env python3
"""Routes API pour l'environnement Reinforcement Learning.

Ce module expose des endpoints REST pour interagir avec l'environnement RL,
notamment pour l'optimisation Optuna des hyperparamètres.
"""

import logging
import threading
from typing import Any

from flask import current_app
from flask_restx import Namespace, Resource, fields

from tasks.rl_tasks import optuna_optimize_impl

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

            # Lancer l'optimisation en arrière-plan (threading)
            def run_optuna_optimization():
                """Fonction exécutée en arrière-plan pour l'optimisation Optuna."""
                try:
                    # Créer un contexte d'application Flask pour le thread
                    with current_app.app_context():
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
