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


@celery.task(
    name="tasks.security_tasks.reap_expired_provisional_device_sessions",
    bind=True,
)
def reap_expired_provisional_device_sessions(self) -> dict[str, int]:  # noqa: ARG001
    """Housekeeping : libère les sessions provisional expirées (tous users).

    Le respect du quota repose sur le reap sync sous FOR UPDATE ; cette tâche
    est un filet de sécurité pour les comptes inactifs.
    """
    app = get_flask_app()
    with app.app_context():
        from ext import db
        from models.mobile_device_session import (
            MobileDeviceSession,
            MobileDeviceSessionStatus,
        )
        from security.mobile_device_session_service import (
            publish_session_revoked,
            revoke_session_state,
        )
        from security.refresh_token_service import revoke_tokens_for_session

        now = datetime.now(UTC)
        expired = (
            MobileDeviceSession.query.filter(
                MobileDeviceSession.status == MobileDeviceSessionStatus.active,
                MobileDeviceSession.confirmed_at.is_(None),
                MobileDeviceSession.provisional_expires_at.isnot(None),
                MobileDeviceSession.provisional_expires_at <= now,
            )
            .limit(500)
            .all()
        )
        revoked_ids = []
        for sess in expired:
            revoke_session_state(
                sess,
                reason="provisional_expired",
                status=MobileDeviceSessionStatus.revoked,
            )
            try:
                revoke_tokens_for_session(
                    str(sess.session_id),
                    reason="provisional_expired",
                    commit=False,
                )
            except Exception as exc:
                logger.warning("reap provisional tokens: %s", exc)
            revoked_ids.append(sess.session_id)
        if revoked_ids:
            db.session.commit()
            for sid in revoked_ids:
                publish_session_revoked(sid)
        logger.info("Reaped %d provisional device sessions", len(revoked_ids))
        return {"reaped_count": len(revoked_ids)}
