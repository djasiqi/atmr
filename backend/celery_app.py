# backend/celery_app.py
from __future__ import annotations
import os
import logging
from celery import Celery
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
    ],
)

# Configure Celery
celery.conf.update(
    timezone=CELERY_TIMEZONE,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_track_started=True,
    task_time_limit=600,  # 10 minutes max per task
    worker_max_tasks_per_child=200,  # Restart worker after 200 tasks
    worker_prefetch_multiplier=1,  # One task at a time
    task_acks_late=True,  # Acknowledge task after execution
    task_reject_on_worker_lost=True,  # Requeue task if worker dies
    broker_connection_retry_on_startup=True,  # important avec Docker
)

# Configure Beat schedule
celery.conf.beat_schedule = {
    "dispatch-autorun": {
        "task": "tasks.dispatch_tasks.autorun_tick",
        "schedule": DISPATCH_AUTORUN_INTERVAL_SEC,
        "options": {"expires": DISPATCH_AUTORUN_INTERVAL_SEC * 2},
    },
    # Planning scans (daily at ~04:15, seconds granularity acceptable in dev)
    "planning-compliance-scan": {
        "task": "planning.compliance_scan",
        "schedule": 24 * 3600,
        "options": {"expires": 6 * 3600},
    },
}

# Initialize Flask app for Celery workers
flask_app = None

def get_flask_app():
    """Get or create Flask app instance for Celery workers."""
    global flask_app
    if flask_app is None:
        from app import create_app
        config_name = os.getenv("FLASK_CONFIG", "production")
        flask_app = create_app(config_name)
        logger.info(f"Flask app created for Celery with config: {config_name}")
    return flask_app


class ContextTask(celery.Task):
    """Custom Celery task that runs within Flask application context."""
    def __call__(self, *args, **kwargs):
        app = get_flask_app()
        with app.app_context():
            return self.run(*args, **kwargs)


# Set the custom task class as default
celery.Task = ContextTask


def init_app(app: Flask) -> Celery:
    """
    Initialize Celery with Flask app context.
    Call this from create_app().
    """
    global flask_app
    flask_app = app
    
    logger.info(
        f"Celery initialized with broker={CELERY_BROKER_URL}, "
        f"backend={CELERY_RESULT_BACKEND}, timezone={CELERY_TIMEZONE}"
    )
    return celery