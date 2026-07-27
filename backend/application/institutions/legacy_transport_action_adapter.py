# pyright: reportImportCycles=false
"""Adaptateur anti-corruption : TransportActionCompleted → événements legacy.

Règle : aucun composant du cœur TransportActionWorkflow ne publie
BookingCancelledEvent directement — uniquement via cet adaptateur.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from models.booking_change_request import BookingChangeRequest

logger = logging.getLogger(__name__)


def publish_legacy_after_transport_action_completed(
    *,
    action: BookingChangeRequest,
    is_cancellation: bool,
) -> None:
    """Traduit Completed vers le fanout historique si nécessaire."""
    if not is_cancellation:
        return

    from domain.events.events import BookingCancelledEvent
    from ext import db
    from models import Booking
    from shared.events.event_bus import publish_event

    booking = db.session.get(Booking, action.booking_id)
    if not booking:
        logger.warning(
            "[LegacyAdapter] booking=%s introuvable pour action=%s",
            action.booking_id,
            action.id,
        )
        return

    company_id = cast(
        int | None, booking.company_id or booking.executing_company_id
    )
    publish_event(
        BookingCancelledEvent(
            booking_id=booking.id,
            driver_id=None,  # déjà cleared en TX
            company_id=company_id,
            actor_role="institution",
            actor_id=action.institution_id,
            cancel_reason=action.reason or action.rejection_reason,
            cancel_source="institution_transport_action",
        )
    )
    logger.info(
        "[LegacyAdapter] BookingCancelledEvent publié via adapter action=%s booking=%s",
        action.id,
        booking.id,
    )
