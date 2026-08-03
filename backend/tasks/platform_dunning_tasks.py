"""Tâches Celery — recouvrement facturation plateforme (art. 6 bis)."""

from __future__ import annotations

import logging

from celery_app import celery

logger = logging.getLogger(__name__)


@celery.task(name="tasks.platform_dunning_tasks.run_platform_dunning_cycle")
def run_platform_dunning_cycle() -> dict:
    """Cycle quotidien : overdue → dossiers → outbox notifications."""
    from services.platform_billing.dunning import run_dunning_cycle

    result = run_dunning_cycle()
    logger.info("platform_dunning_cycle result=%s", result)
    return result
