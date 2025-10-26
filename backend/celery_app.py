# backend/celery_app.py
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

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
    ],
)

# Configure Celery
celery.conf.update(
    timezone=CELERY_TIMEZONE,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_time_limit=0.600,  # 10 minutes max per task
    task_soft_time_limit=0.540,  # 9 minutes soft limit (warning before hard kill)
    worker_max_tasks_per_child=0.200,  # Restart worker after 200 tasks
    worker_prefetch_multiplier=1,  # One task at a time
    task_acks_late=True,  # Acknowledge task after execution
    task_reject_on_worker_lost=True,  # Requeue task if worker dies
    broker_connection_retry_on_startup=True,  # important avec Docker
)

# Configure Beat schedule
celery.conf.beat_schedule = {
    # Dispatch automatique périodique (toutes les 5 min par défaut)
    "dispatch-autorun": {
        "task": "tasks.dispatch_tasks.autorun_tick",
        "schedule": DISPATCH_AUTORUN_INTERVAL_SEC,
        "options": {"expires": DISPATCH_AUTORUN_INTERVAL_SEC * 2},
    },
    # 🆕 Monitoring temps réel pour dispatch autonome (toutes les 2 min)
    "realtime-monitoring": {
        "task": "tasks.dispatch_tasks.realtime_monitoring_tick",
        "schedule": 120.0,  # 2 minutes
        "options": {"expires": 240},  # Expire après 4 min
    },
    # Planning scans (daily at ~04:15, seconds granularity acceptable in dev)
    "planning-compliance-scan": {
        "task": "planning.compliance_scan",
        "schedule": 24 * 3600,
        "options": {"expires": 6 * 3600},
    },
    # 🆕 RL: Ré-entraînement DQN hebdomadaire (dimanche à 3h)
    "rl-retrain-weekly": {
        "task": "tasks.rl_retrain_model",
        "schedule": 7 * 24 * 3600,  # 1 semaine
        "options": {"expires": 12 * 3600},  # Expire après 12h
    },
    # 🆕 RL: Nettoyage feedbacks mensuels (1er du mois à 4h)
    "rl-cleanup-monthly": {
        "task": "tasks.rl_cleanup_old_feedbacks",
        "schedule": 30 * 24 * 3600,  # ~1 mois
        "options": {"expires": 24 * 3600},
    },
    # 🆕 RL: Rapport hebdomadaire (lundi à 8h)
    "rl-weekly-report": {
        "task": "tasks.rl_generate_weekly_report",
        "schedule": 7 * 24 * 3600,  # 1 semaine
        "options": {"expires": 6 * 3600},
    },
}

# Initialize Flask app for Celery workers
_flask_app = {}

def get_flask_app():
    """Get or create Flask app instance for Celery workers."""
    if "app" not in _flask_app:
        from app import create_app
        config_name = os.getenv("FLASK_CONFIG", "production")
        _flask_app["app"] = create_app(config_name)
        logger.info("Flask app created for Celery with config: %s", config_name)
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
        CELERY_BROKER_URL, CELERY_RESULT_BACKEND, CELERY_TIMEZONE
    )
    return celery
