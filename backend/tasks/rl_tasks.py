# backend/tasks/rl_tasks.py
# pyright: reportMissingImports=false

"""Tâches Celery pour le système RL (Reinforcement Learning).

Comprend :
- Ré-entraînement périodique du modèle DQN
- Nettoyage des anciennes métriques
- Génération rapports performance
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from celery_app import celery
from ext import db
from models import RLFeedback

# Constantes
TO_DELETE_ZERO = 0
TOTAL_SUGGESTIONS_ZERO = 0
TOTAL_FEEDBACKS_ZERO = 0
MIN_FEEDBACKS_FOR_TRAINING = 50
MIN_TRAINING_SAMPLES = 30
MAX_STEPS_PER_EPISODE = 200  # Nombre maximum de steps par épisode

logger = logging.getLogger(__name__)


@celery.task(
    name="tasks.rl_retrain_model",
    bind=True,
    acks_late=True,
    task_time_limit=2100,
    task_soft_time_limit=1800,
    max_retries=1,  # 1 retry en cas d'échec transitoire
    autoretry_for=(TimeoutError, ConnectionError),
)
def retrain_dqn_model_task(self):  # noqa: ARG001
    """Tâche Celery : Ré-entraînement hebdomadaire du modèle DQN.

    Exécutée automatiquement chaque dimanche à 3h du matin.

    Steps:
    1. Récupérer feedbacks des 7 derniers jours
    2. Filtrer feedbacks valides pour l'entraînement
    3. Préparer les échantillons d'entraînement
    4. Ré-entraîner le modèle DQN
    5. Sauvegarder le modèle amélioré
    6. Logger les résultats

    Returns:
        dict: Résultat du ré-entraînement

    """
    logger.info("[RL] 🎓 Démarrage ré-entraînement DQN hebdomadaire...")

    try:
        # Récupérer feedbacks dernière semaine
        cutoff = datetime.now(UTC) - timedelta(days=7)
        feedbacks = RLFeedback.query.filter(
            RLFeedback.created_at >= cutoff,
            RLFeedback.suggestion_state.isnot(None),  # Besoin de l'état
        ).all()

        logger.info(
            "[RL] %s feedbacks trouvés dans les 7 derniers jours", len(feedbacks)
        )

        if len(feedbacks) < MIN_FEEDBACKS_FOR_TRAINING:
            logger.warning(
                (
                    "[RL] ⚠️ Pas assez de feedbacks pour ré-entraîner "
                    "(%s/%s minimum). Ré-entraînement reporté."
                ),
                len(feedbacks),
                MIN_FEEDBACKS_FOR_TRAINING,
            )
            return {
                "status": "skipped",
                "reason": "not_enough_feedbacks",
                "feedbacks_count": len(feedbacks),
                "minimum_required": MIN_FEEDBACKS_FOR_TRAINING,
            }

        # Filtrer feedbacks valides pour l'entraînement
        training_samples = []
        for fb in feedbacks:
            if not fb.is_training_ready():
                continue

            # Calculer reward
            reward = fb.calculate_reward()
            if reward is None:
                continue

            training_samples.append(
                {
                    "state": fb.suggestion_state,
                    "action": fb.suggestion_action or 0,
                    "reward": reward,
                    "booking_id": fb.booking_id,
                    "action_taken": fb.action,
                }
            )

        logger.info(
            "[RL] %s échantillons valides pour l'entraînement", len(training_samples)
        )

        if len(training_samples) < MIN_TRAINING_SAMPLES:
            logger.warning(
                (
                    "[RL] ⚠️ Pas assez d'échantillons valides "
                    "(%s/%s minimum). Ré-entraînement reporté."
                ),
                len(training_samples),
                MIN_TRAINING_SAMPLES,
            )
            return {
                "status": "skipped",
                "reason": "not_enough_valid_samples",
                "valid_samples_count": len(training_samples),
                "minimum_required": MIN_TRAINING_SAMPLES,
            }

        # Tentative de ré-entraînement
        try:
            # Importer uniquement si PyTorch disponible
            from services.ml.rl.improved_dqn_agent import ImprovedDQNAgent

            # Charger le modèle actuel
            model_path = "data/rl/models/dqn_best.pth"
            logger.info("[RL] Chargement modèle depuis %s...", model_path)

            try:
                agent = ImprovedDQNAgent.load(  # type: ignore[call-arg]
                    filepath=model_path
                )
            except FileNotFoundError:
                logger.warning(
                    "[RL] ⚠️ Modèle %s introuvable. Création d'un nouveau modèle...",
                    model_path,
                )
                # Créer nouveau modèle
                agent = ImprovedDQNAgent(
                    state_dim=19,  # Match avec suggestion_generator
                    action_dim=5,  # 5 drivers max
                    learning_rate=0.00001,
                )

            # Ré-entraîner avec les échantillons
            logger.info(
                "[RL] Ré-entraînement avec %s échantillons...", len(training_samples)
            )

            total_loss = 0
            for i, sample in enumerate(training_samples):
                # Ajouter à la mémoire de l'agent (méthode générique)
                if hasattr(agent.memory, "add_transition"):
                    agent.memory.add_transition(
                        state=sample["state"],
                        action=sample["action"],
                        reward=sample["reward"],
                        next_state=sample["state"],
                        done=True,
                    )
                elif hasattr(agent.memory, "add"):
                    agent.memory.add(
                        state=sample["state"],
                        action=sample["action"],
                        reward=sample["reward"],
                        next_state=sample["state"],
                        done=True,
                    )

                # Effectuer un pas d'entraînement
                loss = agent.learn()

                if loss is not None:
                    total_loss += loss

                # MAGIC_VALUE_10: logging toujours activé
                logger.debug(
                    "[RL] Échantillon %s/%s traité", i + 1, len(training_samples)
                )

            avg_loss = (
                total_loss / len(training_samples) if len(training_samples) > 0 else 0
            )

            # Sauvegarder le modèle amélioré
            logger.info("[RL] Sauvegarde modèle amélioré vers %s...", model_path)
            agent.save(model_path)

            # Statistiques
            positive_rewards = sum(1 for s in training_samples if s["reward"] > 0)
            negative_rewards = sum(1 for s in training_samples if s["reward"] < 0)
            avg_reward = sum(s["reward"] for s in training_samples) / len(
                training_samples
            )

            result = {
                "status": "success",
                "samples_used": len(training_samples),
                "positive_rewards": positive_rewards,
                "negative_rewards": negative_rewards,
                "avg_reward": round(avg_reward, 2),
                "avg_loss": round(avg_loss, 4),
                "model_path": model_path,
                "timestamp": datetime.now(UTC).isoformat(),
            }

            logger.info(
                (
                    "[RL] ✅ Ré-entraînement réussi ! "
                    "Échantillons: %s, Reward moyen: %s, Loss moyen: %s"
                ),
                len(training_samples),
                avg_reward,
                avg_loss,
            )

            return result

        except ImportError as e:
            logger.warning(
                (
                    "[RL] ⚠️ PyTorch/DQN non disponible dans cet environnement: %s. "
                    "Ré-entraînement impossible. "
                    "Feedbacks sauvegardés pour analyse manuelle."
                ),
                e,
            )
            return {
                "status": "skipped",
                "reason": "pytorch_not_available",
                "message": "Feedbacks sauvegardés pour analyse manuelle",
                "feedbacks_count": len(feedbacks),
                "valid_samples_count": len(training_samples),
            }

    except Exception as e:
        logger.exception("[RL] ❌ Erreur lors du ré-entraînement DQN")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


@celery.task(name="tasks.rl_cleanup_old_feedbacks")
def cleanup_old_feedbacks_task(days_to_keep=90):
    """Tâche Celery : Nettoyer les anciens feedbacks (>90 jours).

    Exécutée mensuellement pour libérer de l'espace DB.

    Args:
        days_to_keep: Nombre de jours de feedbacks à conserver

    Returns:
        dict: Nombre de feedbacks supprimés

    """
    logger.info("[RL] 🧹 Nettoyage feedbacks > %s jours...", days_to_keep)

    try:
        cutoff = datetime.now(UTC) - timedelta(days=days_to_keep)

        # Compter avant suppression
        to_delete = RLFeedback.query.filter(RLFeedback.created_at < cutoff).count()

        if to_delete == TO_DELETE_ZERO:
            logger.info("[RL] Aucun feedback à supprimer")
            return {"status": "success", "deleted_count": 0}

        # Supprimer
        RLFeedback.query.filter(RLFeedback.created_at < cutoff).delete()

        db.session.commit()

        logger.info("[RL] ✅ %s feedbacks supprimés", to_delete)

        return {
            "status": "success",
            "deleted_count": to_delete,
            "cutoff_date": cutoff.isoformat(),
        }

    except Exception as e:
        db.session.rollback()
        logger.exception("[RL] ❌ Erreur lors du nettoyage feedbacks")
        return {"status": "error", "error": str(e)}


@celery.task(name="tasks.rl_generate_weekly_report")
def generate_weekly_report_task():
    """Tâche Celery : Générer rapport hebdomadaire performance RL.

    Exécutée chaque lundi matin pour résumer la semaine précédente.

    Returns:
        dict: Rapport de performance

    """
    logger.info("[RL] 📊 Génération rapport hebdomadaire...")

    try:
        from models import RLFeedback, RLSuggestionMetric

        # Période: 7 derniers jours
        cutoff = datetime.now(UTC) - timedelta(days=7)

        # Statistiques feedbacks
        feedbacks = RLFeedback.query.filter(RLFeedback.created_at >= cutoff).all()

        total_feedbacks = len(feedbacks)
        applied = len([f for f in feedbacks if f.action == "applied"])
        rejected = len([f for f in feedbacks if f.action == "rejected"])

        # Statistiques métriques
        metrics = RLSuggestionMetric.query.filter(
            RLSuggestionMetric.generated_at >= cutoff
        ).all()

        total_suggestions = len(metrics)
        avg_confidence = (
            sum(m.confidence for m in metrics) / total_suggestions
            if total_suggestions > TOTAL_SUGGESTIONS_ZERO
            else 0
        )

        # Précision (si données disponibles)
        metrics_with_actual = [m for m in metrics if m.actual_gain_minutes is not None]
        if metrics_with_actual:
            accuracies = []
            for m in metrics_with_actual:
                acc = m.calculate_gain_accuracy()
                if acc is not None:
                    accuracies.append(acc)
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
        else:
            avg_accuracy = None

        report = {
            "period": "7days",
            "start_date": cutoff.isoformat(),
            "end_date": datetime.now(UTC).isoformat(),
            "suggestions": {
                "total": total_suggestions,
                "avg_confidence": round(avg_confidence, 2),
                "avg_accuracy": round(avg_accuracy, 2) if avg_accuracy else None,
            },
            "feedbacks": {
                "total": total_feedbacks,
                "applied": applied,
                "rejected": rejected,
                "application_rate": round(applied / total_feedbacks, 2)
                if total_feedbacks > TOTAL_FEEDBACKS_ZERO
                else 0,
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

        logger.info(
            "[RL] ✅ Rapport généré : %s suggestions, %s feedbacks, Confiance: %s",
            total_suggestions,
            total_feedbacks,
            avg_confidence,
        )

        return report

    except Exception as e:
        logger.exception("[RL] ❌ Erreur lors de la génération du rapport")
        return {"status": "error", "error": str(e)}


def optuna_optimize_impl(
    company_id: int | None = None,
    data_period: str = "week",
    n_trials: int = 30,
    training_episodes: int = 150,
    eval_episodes: int = 15,
    custom_days: int | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Optimisation Optuna des hyperparamètres DQN (implémentation interne).

    Lance une optimisation Optuna pour trouver les meilleurs hyperparamètres
    du modèle DQN. Les études sont stockées dans PostgreSQL RL et visibles
    dans Optuna Dashboard.

    Cette fonction est utilisée directement par la route admin (via import)
    et peut être appelée par la tâche Celery wrapper.

    Args:
        company_id: ID de l'entreprise (None = toutes les entreprises)
        data_period: Période de données ("week", "month", "custom")
        n_trials: Nombre de trials Optuna à exécuter
        training_episodes: Nombre d'épisodes d'entraînement par trial
        eval_episodes: Nombre d'épisodes d'évaluation par trial
        custom_days: Nombre de jours personnalisé (si data_period="custom")
            Non utilisé pour l'instant, réservé pour usage futur

    Returns:
        dict: Résultat de l'optimisation avec statut et métriques

    """
    import os
    from urllib.parse import quote_plus

    from services.ml.rl.hyperparameter_tuner import HyperparameterTuner

    logger.info(
        (
            "[RL] 🚀 Démarrage optimisation Optuna: "
            "company_id=%s, period=%s, trials=%s, training=%s, eval=%s"
        ),
        company_id,
        data_period,
        n_trials,
        training_episodes,
        eval_episodes,
    )

    try:
        # Construire l'URL PostgreSQL pour Optuna storage
        # Utiliser les variables d'environnement RL
        rl_postgres_user = os.getenv("RL_POSTGRES_USER", "atmr_rl_user")
        rl_postgres_password = os.getenv("RL_POSTGRES_PASSWORD", "atmr_rl_password")
        rl_postgres_host = os.getenv("RL_POSTGRES_HOST", "rl-postgres")
        rl_postgres_port = os.getenv("RL_POSTGRES_PORT", "5432")
        rl_postgres_db = os.getenv("RL_POSTGRES_DB", "atmr_rl_db")

        # Échapper le mot de passe pour l'URL
        password_escaped = quote_plus(rl_postgres_password)

        # Construire l'URL PostgreSQL pour Optuna
        # Format: postgresql://user:password@host:port/database
        optuna_storage = (
            f"postgresql://{rl_postgres_user}:{password_escaped}"
            f"@{rl_postgres_host}:{rl_postgres_port}/{rl_postgres_db}"
        )

        logger.info(
            "[RL] 📊 Storage Optuna configuré: postgresql://%s:***@%s:%s/%s",
            rl_postgres_user,
            rl_postgres_host,
            rl_postgres_port,
            rl_postgres_db,
        )

        # Construire le nom de l'étude (unique par entreprise)
        if company_id:
            study_name = f"dqn_optimization_company_{company_id}"
        else:
            study_name = "dqn_optimization_all_companies"

        # Créer le tuner avec storage PostgreSQL
        tuner = HyperparameterTuner(
            n_trials=n_trials,
            n_training_episodes=training_episodes,
            n_eval_episodes=eval_episodes,
            study_name=study_name,
            storage=optuna_storage,
        )

        # Lancer l'optimisation
        logger.info("[RL] 🎯 Lancement optimisation Optuna...")
        study = tuner.optimize(show_progress_bar=False)

        # ✅ FIX: Sauvegarder automatiquement les meilleurs paramètres
        config_output_path = f"data/rl/optimal_config_{study_name}.json"
        try:
            tuner.save_best_params(study, config_output_path)
            logger.info(
                "[RL] 💾 Meilleurs hyperparamètres sauvegardés: %s", config_output_path
            )
        except Exception as e:
            logger.warning("[RL] ⚠️ Erreur lors de la sauvegarde des paramètres: %s", e)

        # Récupérer les résultats
        best_trial = study.best_trial
        best_value = study.best_value
        n_completed_trials = len(
            [t for t in study.trials if t.state.name == "COMPLETE"]
        )

        result = {
            "status": "success",
            "study_name": study_name,
            "n_trials": n_trials,
            "n_completed_trials": n_completed_trials,
            "best_value": best_value,
            "best_params": best_trial.params if best_trial else None,
            "config_saved_path": config_output_path,
            "company_id": company_id,
            "data_period": data_period,
            "timestamp": datetime.now(UTC).isoformat(),
            "note": (
                f"Optimisation terminée. "
                f"Consultez https://optuna.lirie.ch pour voir les détails de l'étude '{study_name}'. "
                f"Les meilleurs hyperparamètres ont été sauvegardés dans {config_output_path}. "
                f"Utilisez l'endpoint /rl/train/optimal pour entraîner un modèle complet avec ces paramètres."
            ),
        }

        logger.info(
            (
                "[RL] ✅ Optimisation Optuna terminée ! "
                "Study: %s, Best value: %s, Completed trials: %s/%s"
            ),
            study_name,
            best_value,
            n_completed_trials,
            n_trials,
        )

        return result

    except ImportError as e:
        logger.warning("[RL] ⚠️ Optuna/PyTorch non disponible: %s", e)
        return {
            "status": "error",
            "error": "optuna_not_available",
            "message": "Optuna ou PyTorch non disponible dans cet environnement",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception("[RL] ❌ Erreur lors de l'optimisation Optuna")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


# Tâche Celery wrapper (pour exécution via Celery worker si disponible)
@celery.task(
    name="tasks.rl_optuna_optimize",
    bind=True,
    acks_late=True,
    task_time_limit=2100,
    task_soft_time_limit=1800,
    max_retries=0,  # Pas de retry automatique pour les optimisations longues
    autoretry_for=(),
)
def optuna_optimize_task(
    self,  # noqa: ARG001
    company_id: int | None = None,
    data_period: str = "week",
    n_trials: int = 30,
    training_episodes: int = 150,
    eval_episodes: int = 15,
    custom_days: int | None = None,
) -> dict[str, Any]:
    """Wrapper Celery pour optuna_optimize_impl.

    Permet d'exécuter l'optimisation Optuna via un worker Celery.
    Pour usage direct (sans Celery), utiliser optuna_optimize_impl().

    Args:
        self: Binding Celery (requis pour bind=True)
        company_id: ID de l'entreprise (None = toutes les entreprises)
        data_period: Période de données ("week", "month", "custom")
        n_trials: Nombre de trials Optuna à exécuter
        training_episodes: Nombre d'épisodes d'entraînement par trial
        eval_episodes: Nombre d'épisodes d'évaluation par trial
        custom_days: Nombre de jours personnalisé (si data_period="custom")

    Returns:
        dict: Résultat de l'optimisation avec statut et métriques

    """
    # Appeler l'implémentation réelle
    return optuna_optimize_impl(
        company_id=company_id,
        data_period=data_period,
        n_trials=n_trials,
        training_episodes=training_episodes,
        eval_episodes=eval_episodes,
        custom_days=custom_days,
    )


def train_model_with_optimal_params_impl(
    config_path: str | None = None,
    study_name: str | None = None,
    model_output_path: str = "data/rl/models/dqn_optimized.pth",
    training_episodes: int = 1000,
    eval_episodes: int = 50,
    company_id: int | None = None,  # noqa: ARG001
) -> dict[str, Any]:
    """Entraîne un modèle DQN complet avec les hyperparamètres optimaux.

    Cette fonction charge les meilleurs hyperparamètres trouvés par Optuna
    (soit depuis un fichier JSON, soit depuis une étude Optuna),
    puis entraîne un modèle DQN complet avec ces paramètres.

    Args:
        config_path: Chemin vers le fichier optimal_config.json (optionnel)
        study_name: Nom de l'étude Optuna (optionnel, si config_path non fourni)
        model_output_path: Chemin de sortie pour le modèle entraîné
        training_episodes: Nombre d'épisodes d'entraînement complet
        eval_episodes: Nombre d'épisodes d'évaluation finale
        company_id: ID de l'entreprise (réservé pour usage futur)

    Returns:
        dict: Résultat de l'entraînement avec métriques

    """
    import os
    from pathlib import Path
    from urllib.parse import quote_plus

    from services.ml.rl.dispatch_env import DispatchEnv
    from services.ml.rl.improved_dqn_agent import ImprovedDQNAgent

    logger.info(
        (
            "[RL] 🎓 Démarrage entraînement modèle avec hyperparamètres optimaux: "
            "config_path=%s, study_name=%s, episodes=%s"
        ),
        config_path,
        study_name,
        training_episodes,
    )

    try:
        # 1. Charger les hyperparamètres optimaux
        optimal_params: dict[str, Any] | None = None

        if config_path:
            # Charger depuis fichier JSON
            config_file = Path(config_path)
            if not config_file.exists():
                raise FileNotFoundError(
                    f"Fichier de configuration introuvable: {config_path}"
                )

            with config_file.open("r", encoding="utf-8") as f:
                config_data = json.load(f)
                optimal_params = config_data.get("best_params", {})

            logger.info("[RL] ✅ Hyperparamètres chargés depuis %s", config_path)

        elif study_name:
            # Charger depuis étude Optuna
            import optuna

            # Construire l'URL PostgreSQL pour Optuna storage
            rl_postgres_user = os.getenv("RL_POSTGRES_USER", default="atmr_rl_user")
            rl_postgres_password = os.getenv(
                "RL_POSTGRES_PASSWORD", default="atmr_rl_password"
            )
            rl_postgres_host = os.getenv("RL_POSTGRES_HOST", default="rl-postgres")
            rl_postgres_port = os.getenv("RL_POSTGRES_PORT", default="5432")
            rl_postgres_db = os.getenv("RL_POSTGRES_DB", default="atmr_rl_db")

            password_escaped = quote_plus(rl_postgres_password)
            optuna_storage = (
                f"postgresql://{rl_postgres_user}:{password_escaped}"
                f"@{rl_postgres_host}:{rl_postgres_port}/{rl_postgres_db}"
            )

            # Charger l'étude
            study = optuna.load_study(study_name=study_name, storage=optuna_storage)
            optimal_params = study.best_params

            logger.info(
                "[RL] ✅ Hyperparamètres chargés depuis étude Optuna: %s", study_name
            )

        else:
            # Utiliser les hyperparamètres par défaut
            from services.ml.rl.optimal_hyperparameters import OptimalHyperparameters

            optimal_params = OptimalHyperparameters.get_optimal_config("training")
            logger.info("[RL] ✅ Utilisation hyperparamètres par défaut")

        if not optimal_params:
            raise ValueError("Aucun hyperparamètre optimal trouvé")

        # 2. Créer l'environnement avec les paramètres optimaux
        num_drivers = optimal_params.get("num_drivers", 10)
        max_bookings = optimal_params.get("max_bookings", 20)
        env = DispatchEnv(
            num_drivers=num_drivers,
            max_bookings=max_bookings,
            simulation_hours=8,
        )

        # 3. Créer l'agent avec les hyperparamètres optimaux
        agent = ImprovedDQNAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            learning_rate=optimal_params.get("learning_rate", 9.32e-05),
            gamma=optimal_params.get("gamma", 0.951),
            epsilon_start=optimal_params.get("epsilon_start", 0.85),
            epsilon_end=optimal_params.get("epsilon_end", 0.055),
            epsilon_decay=optimal_params.get("epsilon_decay", 0.993),
            batch_size=optimal_params.get("batch_size", 128),
            buffer_size=optimal_params.get("buffer_size", 200000),
            target_update_freq=optimal_params.get("target_update_freq", 13),
            use_double_dqn=optimal_params.get("use_double_dqn", True),
            use_prioritized_replay=optimal_params.get("use_prioritized_replay", True),
            alpha=optimal_params.get("alpha", 0.6),
            beta_start=optimal_params.get("beta_start", 0.4),
            beta_end=optimal_params.get("beta_end", 1.0),
            tau=optimal_params.get("tau", 0.005),
            use_n_step=optimal_params.get("use_n_step", True),
            n_step=optimal_params.get("n_step", 3),
            n_step_gamma=optimal_params.get("n_step_gamma", 0.99),
            use_dueling=optimal_params.get("use_dueling", True),
        )

        logger.info(
            "[RL] 🚀 Démarrage entraînement complet (%s épisodes)...", training_episodes
        )

        # 4. Entraîner le modèle
        episode_rewards = []
        episode_losses = []

        for episode in range(training_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_loss = 0.0
            done = False
            steps = 0

            while not done and steps < MAX_STEPS_PER_EPISODE:
                action = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                agent.store_transition(
                    state, action, reward, next_state, done or truncated
                )

                if len(agent.memory) >= agent.batch_size:
                    loss = agent.learn()
                    episode_loss += loss

                state = next_state
                episode_reward += reward
                steps += 1

            episode_rewards.append(episode_reward)
            if episode_loss > 0.0:
                episode_losses.append(episode_loss / steps if steps > 0 else 0.0)

            # Logging périodique
            if (episode + 1) % 100 == 0:
                recent_avg = sum(episode_rewards[-100:]) / 100
                logger.info(
                    "[RL] Épisode %s/%s - Reward moyen (100 derniers): %.2f",
                    episode + 1,
                    training_episodes,
                    recent_avg,
                )

        # 5. Évaluer le modèle final
        logger.info("[RL] 📊 Évaluation finale (%s épisodes)...", eval_episodes)
        eval_rewards = []
        agent.epsilon = 0.0  # Mode exploitation pure

        for _ in range(eval_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            done = False
            steps = 0

            while not done and steps < MAX_STEPS_PER_EPISODE:
                action = agent.select_action(state)
                next_state, reward, done, truncated, _ = env.step(action)
                state = next_state
                episode_reward += reward
                steps += 1

            eval_rewards.append(episode_reward)

        avg_eval_reward = sum(eval_rewards) / len(eval_rewards)
        std_eval_reward = (
            sum((r - avg_eval_reward) ** 2 for r in eval_rewards) / len(eval_rewards)
        ) ** 0.5

        # 6. Sauvegarder le modèle
        model_path = Path(model_output_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(str(model_path))

        logger.info(
            "[RL] ✅ Modèle sauvegardé: %s (Reward moyen évaluation: %.2f ± %.2f)",
            model_path,
            avg_eval_reward,
            std_eval_reward,
        )

        env.close()

        return {
            "status": "success",
            "model_path": str(model_path),
            "training_episodes": training_episodes,
            "eval_episodes": eval_episodes,
            "avg_training_reward": sum(episode_rewards) / len(episode_rewards),
            "avg_eval_reward": avg_eval_reward,
            "std_eval_reward": std_eval_reward,
            "best_training_reward": max(episode_rewards),
            "worst_training_reward": min(episode_rewards),
            "final_episode_reward": episode_rewards[-1] if episode_rewards else None,
            "hyperparameters_used": optimal_params,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except ImportError as e:
        logger.warning("[RL] ⚠️ PyTorch/Optuna non disponible: %s", e)
        return {
            "status": "error",
            "error": "dependencies_not_available",
            "message": "PyTorch ou Optuna non disponible dans cet environnement",
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.exception("[RL] ❌ Erreur lors de l'entraînement du modèle")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }


# Tâche Celery pour entraîner le modèle avec les hyperparamètres optimaux
@celery.task(
    name="tasks.rl_train_model_optimal",
    bind=True,
    acks_late=True,
    task_time_limit=2100,
    task_soft_time_limit=1800,
    max_retries=0,
    autoretry_for=(),
)
def train_model_with_optimal_params_task(
    self,  # noqa: ARG001
    config_path: str | None = None,
    study_name: str | None = None,
    model_output_path: str = "data/rl/models/dqn_optimized.pth",
    training_episodes: int = 1000,
    eval_episodes: int = 50,
    company_id: int | None = None,
) -> dict[str, Any]:
    """Wrapper Celery pour train_model_with_optimal_params_impl.

    Args:
        self: Binding Celery (requis pour bind=True)
        config_path: Chemin vers optimal_config.json
        study_name: Nom de l'étude Optuna
        model_output_path: Chemin de sortie pour le modèle
        training_episodes: Nombre d'épisodes d'entraînement
        eval_episodes: Nombre d'épisodes d'évaluation
        company_id: ID de l'entreprise

    Returns:
        dict: Résultat de l'entraînement

    """
    return train_model_with_optimal_params_impl(
        config_path=config_path,
        study_name=study_name,
        model_output_path=model_output_path,
        training_episodes=training_episodes,
        eval_episodes=eval_episodes,
        company_id=company_id,
    )
