# backend/tasks/rl_tasks.py

# Constantes pour éviter les valeurs magiques
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

"""Tâches Celery pour le système RL (Reinforcement Learning).

Comprend :
- Ré-entraînement périodique du modèle DQN
- Nettoyage des anciennes métriques
- Génération rapports performance
"""


logger = logging.getLogger(__name__)


@celery.task(name="tasks.rl_retrain_model")
def retrain_dqn_model_task():
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
            from services.rl.improved_dqn_agent import ImprovedDQNAgent

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

    from services.rl.hyperparameter_tuner import HyperparameterTuner

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
            "company_id": company_id,
            "data_period": data_period,
            "timestamp": datetime.now(UTC).isoformat(),
            "note": (
                f"Optimisation terminée. "
                f"Consultez https://optuna.lirie.ch pour voir les détails de l'étude '{study_name}'."
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
    task_time_limit=86400,  # 24 heures max (optimisation longue)
    task_soft_time_limit=82800,  # 23 heures soft limit
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
