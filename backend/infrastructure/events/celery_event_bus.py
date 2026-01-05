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

        celery.send_task("events.handle_domain_event", args=[event.to_dict()])
