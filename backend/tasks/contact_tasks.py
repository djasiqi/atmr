from __future__ import annotations

import logging

from celery_app import celery, get_flask_app

logger = logging.getLogger(__name__)


@celery.task(name="tasks.contact_tasks.retry_failed_contact_notifications")
def retry_failed_contact_notifications_task() -> dict[str, int]:
    app = get_flask_app()
    with app.app_context():
        from services.contact.retry import retry_failed_internal_notifications

        summary = retry_failed_internal_notifications()
        logger.info(
            "[contact_notification_retry] candidates=%s attempted=%s recovered=%s skipped=%s",
            summary.get("candidates"),
            summary.get("attempted"),
            summary.get("recovered"),
            summary.get("skipped"),
        )
        return summary
