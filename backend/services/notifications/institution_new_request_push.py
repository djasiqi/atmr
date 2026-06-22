# services/notifications/institution_new_request_push.py
"""Enqueue push FCM entreprise pour nouvelles offres institution (new_request)."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def enqueue_institution_new_request_company_push(
    *,
    transport_request: Any,
    offer_id: int,
    company_id: int,
    institution_name: str,
    patient_name: str,
    title: str,
    message: str,
    dedupe_key: str,
    mission_date_iso: str | None = None,
    expires_at_iso: str | None = None,
    is_relaunch: bool = False,
    sched: Any | None = None,
) -> None:
    """Construit le message push enrichi et l'enqueue via Celery."""
    try:
        from services.notifications.push_message_builder import (
            build_push_for_institution_new_request,
        )

        msg = build_push_for_institution_new_request(
            transport_request=transport_request,
            offer_id=offer_id,
            company_id=company_id,
            institution_name=institution_name,
            patient_name=patient_name,
            title=title,
            message=message,
            dedupe_key=dedupe_key,
            mission_date_iso=mission_date_iso,
            expires_at_iso=expires_at_iso,
            is_relaunch=is_relaunch,
            sched=sched,
        )

        from tasks.notification_tasks import send_push_company_notification_task

        send_push_company_notification_task.delay(  # pyright: ignore[reportFunctionMemberAccess]
            company_id=company_id,
            title=msg["title"],
            body=msg["body"],
            data=msg["data"],
        )

        try:
            from services.metrics.institution_metrics import (
                track_company_push_new_request_sent,
            )

            track_company_push_new_request_sent(company_id=company_id)
        except Exception:
            logger.debug(
                "[institution_new_request_push] metrics sent failed",
                exc_info=True,
            )

        logger.info(
            "[institution_new_request_push] enqueued company_id=%s offer_id=%s request_id=%s dedupe_key=%s",
            company_id,
            offer_id,
            transport_request.id,
            dedupe_key,
        )
    except Exception as err:
        logger.warning(
            "[institution_new_request_push] enqueue failed company_id=%s offer_id=%s: %s",
            company_id,
            offer_id,
            err,
        )


def enqueue_institution_company_push_message(
    *,
    company_id: int,
    msg: dict[str, Any],
) -> None:
    """Enqueue générique push institution → entreprise (P1 request_updated, etc.)."""
    try:
        from tasks.notification_tasks import send_push_company_notification_task

        send_push_company_notification_task.delay(  # pyright: ignore[reportFunctionMemberAccess]
            company_id=company_id,
            title=str(msg.get("title") or ""),
            body=str(msg.get("body") or ""),
            data=dict(msg.get("data") or {}),
        )
    except Exception as err:
        logger.warning(
            "[institution_company_push] enqueue failed company_id=%s: %s",
            company_id,
            err,
        )
