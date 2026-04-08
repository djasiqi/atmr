"""Tâches Celery — facturation plateforme LIRIE V1."""

from __future__ import annotations

import logging

from celery_app import celery, get_flask_app

logger = logging.getLogger(__name__)


@celery.task(name="platform_billing.recalculate_period")
def recalculate_platform_billing_period_task(period_id: int) -> dict:
    """Recalcul des brouillons pour une période draft (appel async / beat futur)."""
    from services.platform_billing.engine import recalculate_platform_period_drafts

    app = get_flask_app()
    with app.app_context():
        return recalculate_platform_period_drafts(period_id)
