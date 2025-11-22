# backend/celery_app.py
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from celery import Celery

if TYPE_CHECKING:
    from flask import Flask

logger = logging.getLogger(__name__)

# Default configuration (⚠️ par défaut on pointe vers le service Docker 'redis', pas localhost)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
CELERY_TIMEZONE = os.getenv("CELERY_TIMEZONE", "Europe/Zurich")
DISPATCH_AUTORUN_INTERVAL_SEC = int(os.getenv("DISPATCH_AUTORUN_INTERVAL_SEC", "300"))

# Create Celery instance
celery: Celery = Celery(
    "atmr",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        "tasks.dispatch_tasks",
        "tasks.planning_tasks",
        "tasks.rl_tasks",
        "tasks.secret_rotation_tasks",  # ✅ 2.5: Rotation secrets
        "tasks.vault_rotation_tasks",  # ✅ 4.1: Rotation secrets via Vault
        "tasks.purge_tasks",  # ✅ 3.3: Purge données RGPD
        "tasks.profiling_tasks",  # ✅ 3.4: Profiling CPU/mémoire automatique
        "tasks.dlq_cleanup_task",  # ✅ DLQ: Cleanup automatique DLQ
    ],
)

# Configure Celery
celery.conf.update(
    timezone=CELERY_TIMEZONE,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_time_limit=600,  # ✅ 10 minutes max per task (600 secondes) - corrigé de 0.600
    task_soft_time_limit=540,  # ✅ 9 minutes soft limit (540 secondes) - corrigé de 0.540
    worker_max_tasks_per_child=0.200,  # Restart worker after 200 tasks
    worker_prefetch_multiplier=1,  # One task at a time
    task_acks_late=True,  # Acknowledge task after execution
    task_reject_on_worker_lost=True,  # Requeue task if worker dies
    broker_connection_retry_on_startup=True,  # important avec Docker
    # ✅ A3: Configuration des queues (default, realtime, dlq)
    task_routes={
        "tasks.dispatch_tasks.*": {"queue": "default"},
        "tasks.dispatch_tasks.realtime_monitoring_tick": {"queue": "realtime"},
        "tasks.dispatch_tasks.autorun_tick": {"queue": "default"},
        "tasks.planning_tasks.*": {"queue": "default"},
        "tasks.rl_tasks.*": {"queue": "default"},
    },
    task_default_queue="default",
    task_create_missing_queues=True,
    task_default_exchange="default",
    task_default_exchange_type="direct",
    # ✅ DLQ: Configuration pour envoi en DLQ après échecs répétés
    task_default_rate_limit=None,
    task_ignore_result=False,  # Garder les résultats pour debugging
    # ✅ DLQ: Configuration queue DLQ explicite
    task_routes_dlq={
        # Toutes les tâches échouées après max_retries iront en DLQ
        "*": {"queue": "dlq"},
    },
    # Routage automatique vers DLQ après échecs
    task_acks_on_failure_or_timeout=False,  # Ne pas ack si échec (permet DLQ)
)

# Configure Beat schedule
# ✅ 2.6: Jitter ajouté à tous les jobs pour éviter thundering herd
celery.conf.beat_schedule = {
    # Dispatch automatique périodique (toutes les 5 min par défaut)
    "dispatch-autorun": {
        "task": "tasks.dispatch_tasks.autorun_tick",
        "schedule": DISPATCH_AUTORUN_INTERVAL_SEC,
        "options": {
            "expires": DISPATCH_AUTORUN_INTERVAL_SEC * 2,
            "jitter": 30,  # ✅ 2.6: Jitter jusqu'à 30 secondes
        },
    },
    # 🆕 Monitoring temps réel pour dispatch autonome (toutes les 2 min)
    "realtime-monitoring": {
        "task": "tasks.dispatch_tasks.realtime_monitoring_tick",
        "schedule": 120.0,  # 2 minutes
        "options": {
            "expires": 240,  # Expire après 4 min
            "jitter": 15,  # ✅ 2.6: Jitter jusqu'à 15 secondes (tasks fréquentes)
        },
    },
    # Planning scans (daily at ~04:15, seconds granularity acceptable in dev)
    "planning-compliance-scan": {
        "task": "planning.compliance_scan",
        "schedule": 24 * 3600,
        "options": {
            "expires": 6 * 3600,
            "jitter": 300,  # ✅ 2.6: Jitter jusqu'à 5 minutes (tasks quotidiennes)
        },
    },
    # 🆕 RL: Ré-entraînement DQN hebdomadaire (dimanche à 3h)
    "rl-retrain-weekly": {
        "task": "tasks.rl_retrain_model",
        "schedule": 7 * 24 * 3600,  # 1 semaine
        "options": {
            "expires": 12 * 3600,  # Expire après 12h
            "jitter": 1800,  # ✅ 2.6: Jitter jusqu'à 30 minutes (tasks hebdomadaires)
        },
    },
    # 🤖 Agent Dispatch: Démarrer automatiquement tous les agents en mode FULLY_AUTO (toutes les 5 min)
    "ensure-agents-running": {
        "task": "tasks.dispatch_tasks.ensure_agents_running",
        "schedule": 300.0,  # 5 minutes
        "options": {
            "expires": 600,  # Expire après 10 min
            "jitter": 30,  # ✅ 2.6: Jitter jusqu'à 30 secondes
        },
    },
    # 🆕 RL: Nettoyage feedbacks mensuels (1er du mois à 4h)
    "rl-cleanup-monthly": {
        "task": "tasks.rl_cleanup_old_feedbacks",
        "schedule": 30 * 24 * 3600,  # ~1 mois
        "options": {
            "expires": 24 * 3600,
            "jitter": 3600,  # ✅ 2.6: Jitter jusqu'à 1 heure (tasks mensuelles)
        },
    },
    # 🆕 RL: Rapport hebdomadaire (lundi à 8h)
    "rl-weekly-report": {
        "task": "tasks.rl_generate_weekly_report",
        "schedule": 7 * 24 * 3600,  # 1 semaine
        "options": {
            "expires": 6 * 3600,
            "jitter": 1800,  # ✅ 2.6: Jitter jusqu'à 30 minutes (tasks hebdomadaires)
        },
    },
    # ✅ 2.5: Vérification rotation clés (tous les 7 jours pour détecter si rotation due)
    "secret-rotation-check": {
        "task": "tasks.secret_rotation_tasks.check_rotation_due",
        "schedule": 7 * 24 * 3600,  # 1 semaine
        "options": {
            "expires": 12 * 3600,
            "jitter": 3600,  # ✅ 2.6: Jitter jusqu'à 1 heure (tasks hebdomadaires)
        },
    },
    # ✅ 3.3: Purge automatique données RGPD (hebdomadaire)
    "gdpr-purge-all": {
        "task": "tasks.purge_tasks.purge_all_old_data",
        "schedule": 7 * 24 * 3600,  # 1 semaine
        "options": {
            "expires": 24 * 3600,  # Expire après 24h
            "jitter": 3600,  # ✅ 2.6: Jitter jusqu'à 1 heure (tasks hebdomadaires)
        },
    },
    # ✅ 3.3: Anonymisation données utilisateur (mensuelle)
    "gdpr-anonymize-users": {
        "task": "tasks.purge_tasks.anonymize_old_user_data",
        "schedule": 30 * 24 * 3600,  # 1 mois
        "options": {
            "expires": 48 * 3600,  # Expire après 48h
            "jitter": 7200,  # ✅ 2.6: Jitter jusqu'à 2 heures (tasks mensuelles)
        },
    },
    # ✅ 3.4: Profiling automatique hebdomadaire
    "profiling-weekly": {
        "task": "tasks.profiling_tasks.run_weekly_profiling",
        "schedule": 7 * 24 * 3600,  # 1 semaine
        "options": {
            "expires": 24 * 3600,  # Expire après 24h
            "jitter": 3600,  # ✅ 2.6: Jitter jusqu'à 1 heure (tasks hebdomadaires)
        },
    },
    # ✅ 4.1: Rotation automatique JWT secret via Vault (tous les 30 jours)
    "vault-rotate-jwt": {
        "task": "tasks.vault_rotation_tasks.rotate_jwt_secret",
        "schedule": 30 * 24 * 3600,  # 30 jours
        "options": {
            "expires": 48 * 3600,  # Expire après 48h
            "jitter": 7200,  # ✅ 2.6: Jitter jusqu'à 2 heures (tasks mensuelles)
        },
    },
    # ✅ 4.1: Rotation automatique SECRET_KEY Flask via Vault (tous les 90 jours)
    "vault-rotate-flask-secret": {
        "task": "tasks.vault_rotation_tasks.rotate_flask_secret_key",
        "schedule": 90 * 24 * 3600,  # 90 jours
        "options": {
            "expires": 48 * 3600,  # Expire après 48h
            "jitter": 7200,  # ✅ 2.6: Jitter jusqu'à 2 heures (tasks trimestrielles)
        },
    },
    # ✅ 4.1: Rotation automatique encryption key via Vault (tous les 90 jours)
    "vault-rotate-encryption": {
        "task": "tasks.vault_rotation_tasks.rotate_encryption_key",
        "schedule": 90 * 24 * 3600,  # 90 jours
        "options": {
            "expires": 48 * 3600,  # Expire après 48h
            "jitter": 7200,  # ✅ 2.6: Jitter jusqu'à 2 heures (tasks trimestrielles)
        },
    },
    # ✅ 4.1: Rotation globale des secrets (tous les 90 jours, backup de la rotation individuelle)
    "vault-rotate-all": {
        "task": "tasks.vault_rotation_tasks.rotate_all_secrets",
        "schedule": 90 * 24 * 3600,  # 90 jours
        "options": {
            "expires": 48 * 3600,  # Expire après 48h
            "jitter": 7200,  # ✅ 2.6: Jitter jusqu'à 2 heures
        },
    },
    # ✅ DLQ: Cleanup automatique DLQ (quotidien)
    "dlq-cleanup-daily": {
        "task": "tasks.dlq_cleanup_task.cleanup_old_dlq_entries",
        "schedule": 24 * 3600,  # 1 jour
        "options": {
            "expires": 6 * 3600,  # Expire après 6h
            "jitter": 1800,  # ✅ 2.6: Jitter jusqu'à 30 minutes
        },
    },
}

# Initialize Flask app for Celery workers
_flask_app = {}


def get_flask_app():
    """Get or create Flask app instance for Celery workers."""
    if "app" not in _flask_app:
        from app import create_app

        config_name = os.getenv("FLASK_CONFIG", "production")
        # ✅ Désactiver l'initialisation des routes API dans le contexte Celery
        # pour éviter l'erreur Flask-RESTX "View function mapping is overwriting an existing endpoint function: specs"
        original_skip = os.getenv("SKIP_ROUTES_INIT", "false")
        os.environ["SKIP_ROUTES_INIT"] = "true"
        try:
            _flask_app["app"] = create_app(config_name)
            logger.info(
                "Flask app created for Celery with config: %s (routes init skipped)",
                config_name,
            )
        finally:
            # Restaurer la valeur originale
            os.environ["SKIP_ROUTES_INIT"] = original_skip
    return _flask_app["app"]


class ContextTask(celery.Task):
    """Custom Celery task that runs within Flask application context."""

    def __call__(self, *args, **kwargs):
        app = get_flask_app()
        with app.app_context():
            return self.run(*args, **kwargs)


# Set the custom task class as default
celery.Task = ContextTask


def init_app(app: Flask) -> Celery:
    """Initialize Celery with Flask app context.
    Call this from create_app().
    """
    _flask_app["app"] = app

    logger.info(
        "Celery initialized with broker=%s, backend=%s, timezone=%s",
        CELERY_BROKER_URL,
        CELERY_RESULT_BACKEND,
        CELERY_TIMEZONE,
    )

    # ✅ A3: Enregistrer les handlers pour DLQ
    _register_dlq_handlers()

    return celery


# ✅ DLQ: Handler pour stocker métadonnées des tâches échouées avec monitoring
def _register_dlq_handlers():
    """Enregistre les handlers pour capturer les échecs et les stocker en DLQ."""
    from celery.signals import task_failure, task_retry, task_success

    @task_failure.connect
    def task_failed_handler(
        sender=None, task_id=None, exception=None, traceback=None, _einfo=None, **kwargs
    ):  # pyright: ignore[reportUnusedFunction]
        """Gère les tâches échouées après max_retries."""
        task_name = getattr(sender, "name", None) if sender else "unknown"
        retries = kwargs.get("request", {}).get("retries", 0)
        max_retries = getattr(sender, "max_retries", 3) if sender else 3

        # ✅ DLQ: Enregistrer métrique Prometheus
        try:
            from services.unified_dispatch.dispatch_prometheus_metrics import (
                PROMETHEUS_AVAILABLE,
            )

            if PROMETHEUS_AVAILABLE:
                try:
                    from prometheus_client import Counter

                    dlq_counter = Counter(
                        "celery_dlq_failures_total",
                        "Nombre total de tâches envoyées en DLQ",
                        ["task_name"],
                    )
                    dlq_counter.labels(task_name=task_name).inc()
                except ImportError:
                    pass
        except Exception:
            pass  # Ne pas bloquer si métriques échouent

        logger.error(
            "[Celery DLQ] Task %s failed permanently after %d/%d retries: %s",
            task_id,
            retries,
            max_retries,
            str(exception)[:200],
            extra={
                "task_id": task_id,
                "task_name": task_name,
                "exception": str(exception)[:500],
                "traceback": traceback,
                "retries": retries,
                "max_retries": max_retries,
            },
        )

        # ✅ DLQ: Stocker en DB pour visibilité et monitoring
        if task_id:
            try:
                _store_task_failure_in_db(
                    task_id=task_id,
                    task_name=task_name,
                    exception=str(exception),
                    traceback=str(traceback) if traceback else None,
                    args=kwargs.get("args") or [],
                    kwargs=kwargs.get("kwargs") or {},
                )
            except Exception as db_err:
                logger.exception(
                    "[Celery DLQ] Failed to store task failure in DB: %s", db_err
                )

    @task_retry.connect
    def task_retry_handler(_sender=None, task_id=None, reason=None, **kwargs):  # pyright: ignore[reportUnusedFunction]
        """Log les retries."""
        retries = kwargs.get("request", {}).get("retries", 0)
        logger.warning("[Celery] Task %s retry #%d: %s", task_id, retries + 1, reason)

    @task_success.connect
    def task_succeeded_handler(sender=None, **_kwargs):  # pyright: ignore[reportUnusedFunction]
        """Log les succès."""
        logger.debug("[Celery] Task %s succeeded", sender.name if sender else "unknown")


def _store_task_failure_in_db(
    task_id: str,
    task_name: str | None = None,
    exception: str = "",
    traceback: str | None = None,
    args: list[Any] | None = None,
    kwargs: dict[str, Any] | None = None,
):
    """Stocke une tâche échouée dans la table TaskFailure."""
    try:
        from datetime import UTC, datetime

        from ext import db
        from models import TaskFailure

        # Vérifier si la tâche existe déjà (pour éviter doublon)
        existing = TaskFailure.query.filter_by(task_id=task_id).first()

        if existing:
            # Incrémenter le compteur et mettre à jour last_seen
            existing.failure_count += 1
            existing.last_seen = datetime.now(UTC)
            existing.exception = exception[:5000]
            existing.traceback = traceback[:10000] if traceback else None
            db.session.commit()
            logger.info(
                "[Celery DLQ] Updated task failure: task_id=%s (count=%d)",
                task_id,
                existing.failure_count,
            )
        else:
            # Créer nouveau record
            failure_data = {
                "task_id": task_id,
                "task_name": task_name or "unknown",
                "exception": exception[:5000],  # Limiter taille
                "traceback": traceback[:10000] if traceback else None,
                "args": str(args)[:2000] if args else None,
                "kwargs": kwargs if kwargs else None,  # JSONB accepte dict directement
                "first_seen": datetime.now(UTC),
                "last_seen": datetime.now(UTC),
                "failure_count": 1,
            }
            failure = TaskFailure(**failure_data)

            db.session.add(failure)
            db.session.commit()

            logger.info(
                "[Celery DLQ] Stored new task failure: task_id=%s, task_name=%s",
                task_id,
                task_name,
            )

    except Exception as e:
        logger.exception("[Celery DLQ] Failed to store failure in DB: %s", e)
        # Continue sans erreur critique (ne bloque pas le worker)
