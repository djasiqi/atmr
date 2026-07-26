"""Worker Celery F-02 — reprise file tracking_derived_repair_pending."""

from __future__ import annotations

import logging
import os

from celery import shared_task

from celery_app import get_flask_app

logger = logging.getLogger(__name__)


@shared_task(
    name="tasks.tracking_repair_tasks.process_derived_repairs",
    acks_late=True,
    soft_time_limit=50,
    time_limit=60,
)
def process_derived_repairs(limit: int | None = None) -> dict[str, int]:
    """Reprend les réparations Redis canonical en attente."""
    app = get_flask_app()
    lim = limit
    if lim is None:
        lim = int(os.getenv("TRACKING_DERIVED_REPAIR_BATCH_SIZE", "50"))
    with app.app_context():
        from services.tracking.ingest_durability import process_pending_repairs

        result = process_pending_repairs(limit=lim)
        logger.info(
            "[tracking_repair] processed=%s done=%s failed=%s",
            result.get("processed"),
            result.get("done"),
            result.get("failed"),
        )
        return result
