from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from services.contact.dispatcher import get_destination_email


def apply_contact_notification_result(
    row: Any,
    result: dict[str, Any],
    *,
    increment_retry: bool = False,
) -> None:
    """Met à jour les statuts de livraison sans jamais supprimer la demande."""
    row.notification_last_attempt_at = datetime.now(UTC)
    destination = result.get("destination") or get_destination_email()
    row.assigned_channel = destination

    if result.get("internal_ok") or (result.get("ok") and "internal_ok" not in result):
        row.email_delivery_status = "sent"
        row.notification_last_error = None
        if getattr(row, "status", None) == "new":
            row.status = "triaged"
    else:
        row.email_delivery_status = "failed"
        error = str(
            result.get("internal_error")
            or result.get("error")
            or "notification_interne_echec"
        )
        row.notification_last_error = error[:512]
        if increment_retry:
            row.notification_retry_count = (
                int(getattr(row, "notification_retry_count", 0) or 0) + 1
            )
        if getattr(row, "status", None) != "spam":
            row.status = "new"

    if result.get("auto_reply_skipped"):
        row.autoreply_delivery_status = "skipped"
    elif "auto_reply_ok" in result:
        row.autoreply_delivery_status = (
            "sent" if result.get("auto_reply_ok") else "failed"
        )
