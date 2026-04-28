"""Notifications portail client (fin de course) : jalons visibles, sans spam fin de mission."""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger(__name__)

_EndClientMilestone = Literal["company_accepted", "driver_assigned", "en_route"]


def _client_user_public_id_for_booking(booking: Any) -> str | None:  # noqa: PLR0911
    from models import Client, User

    cid = getattr(booking, "client_id", None)
    if cid is None:
        return None
    try:
        client = Client.query.get(int(cid))
    except (TypeError, ValueError):
        return None
    if client is None:
        return None
    uid = getattr(client, "user_id", None)
    if uid is None:
        return None
    try:
        user = User.query.get(int(uid))
    except (TypeError, ValueError):
        return None
    if user is None:
        return None
    pid = getattr(user, "public_id", None)
    if not pid:
        return None
    return str(pid).strip() or None


def _milestone_copy(milestone: _EndClientMilestone) -> tuple[str, str]:
    if milestone == "company_accepted":
        return (
            "Transport confirmé",
            "Une entreprise a accepté votre demande de transport.",
        )
    if milestone == "driver_assigned":
        return (
            "Chauffeur désigné",
            "Un chauffeur a été assigné à votre course.",
        )
    return (
        "Chauffeur en route",
        "Le chauffeur est en route vers le lieu de prise en charge.",
    )


def notify_end_client_booking_milestone(
    booking: Any,
    *,
    milestone: _EndClientMilestone,
    send_push: bool = True,
) -> None:
    """Émet ``client_booking_updated`` (room utilisateur) + push Expo si token présent."""
    public_id = _client_user_public_id_for_booking(booking)
    if not public_id:
        return

    title, body = _milestone_copy(milestone)
    booking_id = int(getattr(booking, "id", 0) or 0)
    try:
        payload: dict[str, Any] = {
            "milestone": milestone,
            "booking_id": booking_id,
            "title": title,
            "body": body,
        }
        if hasattr(booking, "serialize"):
            try:
                payload["booking"] = booking.serialize
            except Exception:
                payload["booking"] = {"id": booking_id}
        else:
            payload["booking"] = {"id": booking_id}

        from services.realtime.socketio import emit_client_user_event

        emit_client_user_event(public_id, "client_booking_updated", payload)
    except Exception:
        logger.exception(
            "[end_client_booking_notify] socket emit failed booking_id=%s",
            booking_id,
        )

    if not send_push:
        return

    try:
        from models import Client, User

        cid = getattr(booking, "client_id", None)
        if cid is None:
            return
        client = Client.query.get(int(cid))
        if client is None or not getattr(client, "user_id", None):
            return
        user = User.query.get(int(client.user_id))
        if user is None:
            return
        token = (getattr(user, "push_token", None) or "").strip()
        if not token:
            return

        from services.notifications.push import send_push_message

        send_push_message(
            token,
            title,
            body,
            {
                "type": "client_booking_updated",
                "milestone": milestone,
                "booking_id": booking_id,
            },
            driver_id=None,
            bypass_rate_limit=False,
        )
    except Exception:
        logger.warning(
            "[end_client_booking_notify] push failed booking_id=%s (non-critical)",
            booking_id,
            exc_info=True,
        )
