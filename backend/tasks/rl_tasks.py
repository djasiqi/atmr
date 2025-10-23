# backend/tasks/rl_tasks.py
# ruff: noqa: DTZ003, W293
"""
Tâches Celery pour le système RL (Reinforcement Learning).

Comprend :
- Ré-entraînement périodique du modèle DQN
- Nettoyage des anciennes métriques
- Génération rapports performance
"""
import logging
from datetime import UTC, datetime, timedelta

from celery_app import celery
from ext import db

logger = logging.getLogger(__name__)


@celery.task(name='tasks.rl_retrain_model')
def retrain_dqn_model_task():
    """
    Tâche Celery : Ré-entraînement hebdomadaire du modèle DQN.
    
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
        from models import RLFeedback

        # Récupérer feedbacks dernière semaine
        cutoff = datetime.now(UTC) - timedelta(days=7)
        feedbacks = RLFeedback.query.filter(
            RLFeedback.created_at >= cutoff,
            RLFeedback.suggestion_state.isnot(None)  # Besoin de l'état
        ).all()

        logger.info(f"[RL] {len(feedbacks)} feedbacks trouvés dans les 7 derniers jours")

        if len(feedbacks) < 50:
            logger.warning(
                f"[RL] ⚠️ Pas assez de feedbacks pour ré-entraîner ({len(feedbacks)}/50 minimum). "
                "Ré-entraînement reporté."
            )
            return {
                "status": "skipped",
                "reason": "not_enough_feedbacks",
                "feedbacks_count": len(feedbacks),
                "minimum_required": 50
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

            training_samples.append({
                'state': fb.suggestion_state,
                'action': fb.suggestion_action or 0,
                'reward': reward,
                'booking_id': fb.booking_id,
                'action_taken': fb.action
            })

        logger.info(f"[RL] {len(training_samples)} échantillons valides pour l'entraînement")

        if len(training_samples) < 30:
            logger.warning(
                f"[RL] ⚠️ Pas assez d'échantillons valides ({len(training_samples)}/30 minimum). "
                "Ré-entraînement reporté."
            )
            return {
                "status": "skipped",
                "reason": "not_enough_valid_samples",
                "valid_samples_count": len(training_samples),
                "minimum_required": 30
            }

        # Tentative de ré-entraînement
        try:
            # Importer uniquement si PyTorch disponible
            from services.rl.dqn_agent import DQNAgent

            # Charger le modèle actuel
            model_path = "data/rl/models/dqn_best.pth"
            logger.info(f"[RL] Chargement modèle depuis {model_path}...")

            try:
                agent = DQNAgent.load(model_path)
            except FileNotFoundError:
                logger.warning(f"[RL] ⚠️ Modèle {model_path} introuvable. Création d'un nouveau modèle...")
                # Créer nouveau modèle
                agent = DQNAgent(
                    state_size=19,  # Match avec suggestion_generator
                    action_size=5,   # 5 drivers max
                    learning_rate=0.0001
                )

            # Ré-entraîner avec les échantillons
            logger.info(f"[RL] Ré-entraînement avec {len(training_samples)} échantillons...")

            total_loss = 0.0
            for i, sample in enumerate(training_samples):
                # Ajouter à la mémoire de l'agent
                agent.memory.push(
                    state=sample['state'],
                    action=sample['action'],
                    next_state=sample['state'],  # État final = état initial (simplification)
                    reward=sample['reward'],
                    done=True
                )
                
                # Effectuer un pas d'entraînement
                loss = agent.train_step()

                if loss is not None:
                    total_loss += loss

                if (i + 1) % 10 == 0:
                    logger.debug(f"[RL] Échantillon {i+1}/{len(training_samples)} traité")

            avg_loss = total_loss / len(training_samples) if len(training_samples) > 0 else 0.0

            # Sauvegarder le modèle amélioré
            logger.info(f"[RL] Sauvegarde modèle amélioré vers {model_path}...")
            agent.save(model_path)

            # Statistiques
            positive_rewards = sum(1 for s in training_samples if s['reward'] > 0)
            negative_rewards = sum(1 for s in training_samples if s['reward'] < 0)
            avg_reward = sum(s['reward'] for s in training_samples) / len(training_samples)

            result = {
                "status": "success",
                "samples_used": len(training_samples),
                "positive_rewards": positive_rewards,
                "negative_rewards": negative_rewards,
                "avg_reward": round(avg_reward, 2),
                "avg_loss": round(avg_loss, 4),
                "model_path": model_path,
                "timestamp": datetime.now(UTC).isoformat()
            }

            logger.info(
                f"[RL] ✅ Ré-entraînement réussi ! "
                f"Échantillons: {len(training_samples)}, "
                f"Reward moyen: {avg_reward:.2f}, "
                f"Loss moyen: {avg_loss:.4f}"
            )

            return result

        except ImportError as e:
            logger.warning(
                f"[RL] ⚠️ PyTorch/DQN non disponible dans cet environnement: {e}. "
                "Ré-entraînement impossible. Feedbacks sauvegardés pour analyse manuelle."
            )
            return {
                "status": "skipped",
                "reason": "pytorch_not_available",
                "message": "Feedbacks sauvegardés pour analyse manuelle",
                "feedbacks_count": len(feedbacks),
                "valid_samples_count": len(training_samples)
            }

    except Exception as e:
        logger.exception("[RL] ❌ Erreur lors du ré-entraînement DQN")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat()
        }


@celery.task(name='tasks.rl_cleanup_old_feedbacks')
def cleanup_old_feedbacks_task(days_to_keep=90):
    """
    Tâche Celery : Nettoyer les anciens feedbacks (>90 jours).
    
    Exécutée mensuellement pour libérer de l'espace DB.
    
    Args:
        days_to_keep: Nombre de jours de feedbacks à conserver
    
    Returns:
        dict: Nombre de feedbacks supprimés
    """
    logger.info(f"[RL] 🧹 Nettoyage feedbacks > {days_to_keep} jours...")

    try:
        from models import RLFeedback

        cutoff = datetime.now(UTC) - timedelta(days=days_to_keep)

        # Compter avant suppression
        to_delete = RLFeedback.query.filter(
            RLFeedback.created_at < cutoff
        ).count()

        if to_delete == 0:
            logger.info("[RL] Aucun feedback à supprimer")
            return {"status": "success", "deleted_count": 0}

        # Supprimer
        RLFeedback.query.filter(
            RLFeedback.created_at < cutoff
        ).delete()

        db.session.commit()

        logger.info(f"[RL] ✅ {to_delete} feedbacks supprimés")

        return {
            "status": "success",
            "deleted_count": to_delete,
            "cutoff_date": cutoff.isoformat()
        }

    except Exception as e:
        db.session.rollback()
        logger.exception("[RL] ❌ Erreur lors du nettoyage feedbacks")
        return {
            "status": "error",
            "error": str(e)
        }


@celery.task(name='tasks.rl_generate_weekly_report')
def generate_weekly_report_task():
    """
    Tâche Celery : Générer rapport hebdomadaire performance RL.
    
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
        feedbacks = RLFeedback.query.filter(
            RLFeedback.created_at >= cutoff
        ).all()

        total_feedbacks = len(feedbacks)
        applied = len([f for f in feedbacks if f.action == 'applied'])
        rejected = len([f for f in feedbacks if f.action == 'rejected'])

        # Statistiques métriques
        metrics = RLSuggestionMetric.query.filter(
            RLSuggestionMetric.generated_at >= cutoff
        ).all()

        total_suggestions = len(metrics)
        avg_confidence = sum(m.confidence for m in metrics) / total_suggestions if total_suggestions > 0 else 0.0

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
            "period": "7_days",
            "start_date": cutoff.isoformat(),
            "end_date": datetime.now(UTC).isoformat(),
            "suggestions": {
                "total": total_suggestions,
                "avg_confidence": round(avg_confidence, 2),
                "avg_accuracy": round(avg_accuracy, 2) if avg_accuracy else None
            },
            "feedbacks": {
                "total": total_feedbacks,
                "applied": applied,
                "rejected": rejected,
                "application_rate": round(applied / total_feedbacks, 2) if total_feedbacks > 0 else 0.0
            },
            "timestamp": datetime.now(UTC).isoformat()
        }

        logger.info(
            f"[RL] ✅ Rapport généré : {total_suggestions} suggestions, "
            f"{total_feedbacks} feedbacks, "
            f"Confiance: {avg_confidence:.0%}"
        )

        return report

    except Exception as e:
        logger.exception("[RL] ❌ Erreur lors de la génération du rapport")
        return {
            "status": "error",
            "error": str(e)
        }

