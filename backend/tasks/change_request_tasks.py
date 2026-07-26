# tasks/change_request_tasks.py
# pyright: reportCallIssue=false, reportArgumentType=false
"""Tâches Celery : expiration des demandes de validation de modification (PR2).

Une BookingChangeRequest PENDING dont `expires_at` est dépassé devient EXPIRED.
Le booking concerné passe en `escalation_required` (action institution requise),
SAUF si AUTO_REFUSE_EXPIRED_CHANGE_REQUESTS=true : dans ce cas, on traite
l'expiration comme un refus transporteur (libération + rediffusion).
"""

from __future__ import annotations

import contextlib
import logging
import os
from datetime import UTC, datetime
from typing import Any

from celery_app import celery
from ext import db
from models import Booking, BookingChangeRequest
from models.booking_change_request import (
    BookingChangeRequestStatus,
    TransportActionEffectStatus,
    TransportActionNextActor,
    TransportActionStatus,
)
from security.audit_log import AuditLogger

logger = logging.getLogger(__name__)


def _auto_refuse_enabled() -> bool:
    """Refus automatique à l'expiration (désactivé par défaut)."""
    return os.getenv("AUTO_REFUSE_EXPIRED_CHANGE_REQUESTS", "false").lower() in (
        "1",
        "true",
        "yes",
    )


@celery.task(
    name="tasks.change_request_tasks.expire_pending_change_requests",
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    autoretry_for=(Exception,),
    queue="default",
)
def expire_pending_change_requests(_self: Any) -> dict[str, int]:
    """Traite les demandes de validation expirées.

    pending -> expired -> escalation_required (ou auto-refus si flag activé).
    """
    logger.info("[ChangeRequestTask] Starting expire_pending_change_requests")
    now = datetime.now(UTC)

    try:
        expired = BookingChangeRequest.query.filter(
            BookingChangeRequest.status.in_(list(TransportActionStatus.OPEN)),
            BookingChangeRequest.expires_at.isnot(None),
            BookingChangeRequest.expires_at < now,
        ).all()

        logger.info(
            "[ChangeRequestTask] Found %d expired change requests", len(expired)
        )

        auto_refuse = _auto_refuse_enabled()
        escalated = 0
        refused = 0

        for change_request in expired:
            try:
                if auto_refuse:
                    _auto_refuse_change_request(change_request)
                    refused += 1
                else:
                    _escalate_change_request(change_request, now)
                    escalated += 1
                db.session.commit()
            except Exception:
                logger.exception(
                    "[ChangeRequestTask] Error processing change_request %s",
                    change_request.id,
                )
                db.session.rollback()
                continue

        logger.info(
            "[ChangeRequestTask] expired=%d escalated=%d auto_refused=%d",
            len(expired),
            escalated,
            refused,
        )
        return {
            "expired": len(expired),
            "escalated": escalated,
            "auto_refused": refused,
        }
    except Exception:
        logger.exception("[ChangeRequestTask] Error in expire_pending_change_requests")
        db.session.rollback()
        raise


def _escalate_change_request(
    change_request: BookingChangeRequest, now: datetime
) -> None:
    """V1 : expiration clôture l'action entière ; mission inchangée."""
    change_request.status = BookingChangeRequestStatus.EXPIRED
    change_request.effect_status = TransportActionEffectStatus.NONE
    change_request.next_actor_type = TransportActionNextActor.NONE
    change_request.version = int(change_request.version or 1) + 1
    change_request.updated_at = now

    booking = Booking.query.get(change_request.booking_id)
    if booking:
        from application.institutions.transport_action_workflow import (
            clear_active_change_request_refs,
        )

        clear_active_change_request_refs(change_request.id)
        booking.updated_at = now

    _record_timeline(change_request, "change_expired")

    with contextlib.suppress(Exception):
        AuditLogger.log_action(
            action_type="transport_action_expired",
            action_category="institution",
            institution_id=change_request.institution_id,
            result_status="success",
            action_details={
                "change_request_id": change_request.id,
                "booking_id": change_request.booking_id,
                "mission_unchanged": True,
            },
        )


def _auto_refuse_change_request(change_request: BookingChangeRequest) -> None:
    """Expiration = clôture action, mission inchangée (pas d'apply / redispatch)."""
    now = datetime.now(UTC)
    booking = Booking.query.get(change_request.booking_id)

    change_request.status = BookingChangeRequestStatus.EXPIRED
    change_request.effect_status = TransportActionEffectStatus.NONE
    change_request.next_actor_type = TransportActionNextActor.NONE
    change_request.responded_by_role = "system"
    change_request.responded_at = now
    change_request.version = int(change_request.version or 1) + 1
    change_request.updated_at = now
    if booking:
        from application.institutions.transport_action_workflow import (
            clear_active_change_request_refs,
        )

        clear_active_change_request_refs(change_request.id)
        booking.updated_at = now
    db.session.flush()

    _record_timeline(change_request, "change_expired")

    with contextlib.suppress(Exception):
        AuditLogger.log_action(
            action_type="transport_action_expired",
            action_category="institution",
            institution_id=change_request.institution_id,
            result_status="success",
            action_details={
                "change_request_id": change_request.id,
                "booking_id": change_request.booking_id,
                "mission_unchanged": True,
            },
        )

    with contextlib.suppress(Exception):
        AuditLogger.log_action(
            action_type="change_request_auto_refused",
            action_category="institution",
            institution_id=change_request.institution_id,
            result_status="success",
            action_details={
                "change_request_id": change_request.id,
                "booking_id": change_request.booking_id,
            },
        )


def _record_timeline(change_request: BookingChangeRequest, event_type: str) -> None:
    """Historise une transition d'expiration dans la timeline transport."""
    with contextlib.suppress(Exception):
        from services.institutions.transport_timeline_service import (
            TimelineActor,
            record_event,
        )

        record_event(
            event_type,
            institution_id=change_request.institution_id,
            transport_request_id=change_request.transport_request_id,
            booking_id=change_request.booking_id,
            actor=TimelineActor(actor_type="system"),
            payload={"change_request_id": change_request.id},
            correlation_id=f"{event_type}:{change_request.id}",
        )
