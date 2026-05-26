from __future__ import annotations

# pyright: reportImplicitOverride=false

try:
    from typing import (
        override,  # Python 3.12+
    )
except ImportError:  # pragma: no cover
    from typing_extensions import override  # Python < 3.12

from application.events.event_bus import EventBus
from domain.events.base import DomainEvent


class CeleryEventBus(EventBus):
    """Bus asynchrone via Celery.

    Envoie les événements vers la tâche `events.handle_domain_event`.
    Cette tâche délègue ensuite au registry (`services/event_handlers_registry.py`).
    """

    @override
    def publish(self, event: DomainEvent) -> None:
        from celery_app import celery

        event_dict = event.to_dict()
        if getattr(event, "event_type", None) == "DriverNewBookingEvent":
            from services.notifications.push_pipeline_log import log_driver_push_stage

            log_driver_push_stage(
                "driver_push.publish",
                event_id=event_dict.get("event_id"),
                correlation_id=event_dict.get("correlation_id"),
                booking_id=event_dict.get("booking_id"),
                driver_id=event_dict.get("driver_id"),
            )

        celery.send_task("events.handle_domain_event", args=[event_dict])
