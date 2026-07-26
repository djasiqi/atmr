"""Tâches Celery pour emails institution (invitation + notification d'accès)."""

from __future__ import annotations

import logging
from typing import Any

from celery_app import celery

logger = logging.getLogger(__name__)


@celery.task(
    name="tasks.institution_invitation_tasks.send_institution_email",
    bind=True,
    acks_late=True,
    task_time_limit=30,
    task_soft_time_limit=20,
    max_retries=3,
    autoretry_for=(ConnectionError, TimeoutError, OSError),
    default_retry_delay=5,
    retry_backoff=True,
)
def send_institution_email_task(
    _self,
    *,
    email_type: str,
    to_email: str,
    first_name: str | None,
    institution_name: str,
    inviter_name: str,
    role: str,
    raw_token: str | None = None,
    user_id: int | None = None,
) -> dict[str, Any]:
    from celery_app import get_flask_app

    app = get_flask_app()
    with app.app_context():
        try:
            from application.institutions.invitation_service import (
                send_institution_access_email,
                send_invitation_email,
            )

            if email_type == "access_notification":
                result = send_institution_access_email(
                    to_email=to_email,
                    first_name=first_name,
                    institution_name=institution_name,
                    inviter_name=inviter_name,
                    role=role,
                )
            else:
                if not raw_token:
                    raise ValueError("raw_token required for invitation email")
                result = send_invitation_email(
                    to_email=to_email,
                    first_name=first_name,
                    institution_name=institution_name,
                    inviter_name=inviter_name,
                    role=role,
                    raw_token=raw_token,
                )

            try:
                from security.institution_metrics import institution_invitations_total

                institution_invitations_total.labels(
                    path="celery",
                    email_type=email_type,
                    result="sent" if result.success else "failed",
                ).inc()
            except Exception:
                pass

            if not result.success:
                raise RuntimeError(result.error or "email_send_failed")

            return {"ok": True, "user_id": user_id, "email_type": email_type}
        except Exception as e:
            logger.exception(
                "[institution_invitation_task] failed email_type=%s user_id=%s: %s",
                email_type,
                user_id,
                e,
            )
            try:
                from security.institution_metrics import institution_invitations_total

                institution_invitations_total.labels(
                    path="celery",
                    email_type=email_type,
                    result="failed",
                ).inc()
            except Exception:
                pass
            raise
