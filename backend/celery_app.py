# backend/celery_app.py
from __future__ import annotations

import os
import logging
from celery import Celery
from celery.schedules import crontab
from flask import Flask

logger = logging.getLogger(__name__)

# Default configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)
CELERY_TIMEZONE = os.getenv("CELERY_TIMEZONE", "Europe/Zurich")
DISPATCH_AUTORUN_INTERVAL_SEC = int(os.getenv("DISPATCH_AUTORUN_INTERVAL_SEC", "300"))

# Create Celery instance
celery = Celery(
    "atmr",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["tasks.dispatch_tasks"],
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
)

# Configure Beat schedule
celery.conf.beat_schedule = {
    "dispatch-autorun": {
        "task": "tasks.dispatch_tasks.autorun_tick",
        "schedule": DISPATCH_AUTORUN_INTERVAL_SEC,
        "options": {"expires": DISPATCH_AUTORUN_INTERVAL_SEC * 2},
    },
}


def init_app(app: Flask) -> None:
    """
    Initialize Celery with Flask app context.
    Call this from create_app().
    """
    class FlaskTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = FlaskTask
    logger.info(
        f"Celery initialized with broker={CELERY_BROKER_URL}, "
        f"backend={CELERY_RESULT_BACKEND}, timezone={CELERY_TIMEZONE}"
    )
    return celery