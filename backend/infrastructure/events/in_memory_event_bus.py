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


class InMemoryEventBus(EventBus):
    """Bus synchrone (tests/dev) qui exécute directement les handlers.

    Utile pour:
        - exécuter les handlers sans worker Celery
        - écrire des tests unitaires simples (`tests/services/test_event_bus.py`)
    """

    @override
    def publish(self, event: DomainEvent) -> None:
        from services.event_handlers_registry import dispatch_event

        dispatch_event(event.to_dict())
