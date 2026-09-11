from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from ext import db
from models import ContactRequest
from services.contact.dispatcher import (
    payload_from_contact_request,
    send_internal_contact_notification,
)
from services.contact.status import apply_contact_notification_result

logger = logging.getLogger(__name__)

MAX_NOTIFICATION_RETRIES = 5
RETRY_COOLDOWN = timedelta(minutes=5)


def _eligible_for_retry(row: ContactRequest, *, ignore_cooldown: bool = False) -> bool:
    if str(row.email_delivery_status or "") != "failed":
        return False
    if str(row.status or "") == "spam":
        return False
    if int(row.notification_retry_count or 0) >= MAX_NOTIFICATION_RETRIES:
        return False
    if ignore_cooldown:
        return True
    last_attempt = row.notification_last_attempt_at
    if last_attempt is None:
        return True
    if last_attempt.tzinfo is None:
        last_attempt = last_attempt.replace(tzinfo=UTC)
    return datetime.now(UTC) - last_attempt >= RETRY_COOLDOWN


def retry_internal_notification(
    row: ContactRequest, *, ignore_cooldown: bool = False
) -> dict[str, Any]:
    if not _eligible_for_retry(row, ignore_cooldown=ignore_cooldown):
        return {
            "ok": False,
            "skipped": True,
            "internal_ok": False,
            "error": "retry_not_eligible",
        }

    payload = payload_from_contact_request(row)
    result = send_internal_contact_notification(payload)
    apply_contact_notification_result(
        row,
        {
            "ok": result.get("ok"),
            "internal_ok": result.get("ok"),
            "internal_error": result.get("error"),
            "destination": result.get("destination"),
            "from_email": result.get("from_email"),
        },
        increment_retry=not bool(result.get("ok")),
    )
    return result


def retry_failed_internal_notifications(*, limit: int = 20) -> dict[str, int]:
    rows = (
        ContactRequest.query.filter(ContactRequest.email_delivery_status == "failed")
        .filter(ContactRequest.status != "spam")
        .filter(ContactRequest.notification_retry_count < MAX_NOTIFICATION_RETRIES)
        .order_by(ContactRequest.created_at.asc())
        .limit(limit)
        .all()
    )
    attempted = 0
    recovered = 0
    skipped = 0
    for row in rows:
        result = retry_internal_notification(row)
        if result.get("skipped"):
            skipped += 1
            continue
        attempted += 1
        if result.get("ok"):
            recovered += 1
            logger.info(
                "contact_internal_notification_retry_ok trace_id=%s",
                row.trace_id,
            )
        else:
            logger.warning(
                "contact_internal_notification_retry_failed trace_id=%s error=%s",
                row.trace_id,
                result.get("error"),
            )
    if attempted or skipped:
        db.session.commit()
    return {
        "candidates": len(rows),
        "attempted": attempted,
        "recovered": recovered,
        "skipped": skipped,
    }
