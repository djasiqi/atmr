"""Taches Celery pour la securite : purge des audit logs expires."""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime, timedelta

from celery_app import celery, get_flask_app

logger = logging.getLogger(__name__)

RETENTION_DAYS = int(os.environ.get("AUDIT_LOG_RETENTION_DAYS", "730"))


@celery.task(name="tasks.security_tasks.purge_expired_audit_logs", bind=True)
def purge_expired_audit_logs(self) -> dict[str, int]:  # noqa: ARG001
    """Supprime les audit logs plus anciens que RETENTION_DAYS."""
    app = get_flask_app()
    with app.app_context():
        from ext import db
        from security.audit_log import AuditLog

        cutoff = datetime.now(UTC) - timedelta(days=RETENTION_DAYS)
        count = AuditLog.query.filter(AuditLog.created_at < cutoff).delete(
            synchronize_session=False
        )
        db.session.commit()
        logger.info("Purged %d audit logs older than %d days", count, RETENTION_DAYS)
        return {"purged_count": count, "retention_days": RETENTION_DAYS}
