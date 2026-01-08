# pyright: reportImportCycles=false
# basedpyright: reportImportCycles=false
# pyright: reportImplicitOverride=false
# Note: basedpyright ne reconnaît pas toujours @override avec imports
# conditionnels (pattern utilisé ailleurs).
from __future__ import annotations

from abc import ABC, abstractmethod

from domain.events.base import DomainEvent

try:
    from typing import (
        override,  # Python 3.12+
    )
except ImportError:  # pragma: no cover
    from typing_extensions import override  # Python < 3.12


class EventBus(ABC):
    """Interface du bus d'événements.

    Un `EventBus` publie des `DomainEvent` sans que la couche Application n'ait à
    connaître l'infrastructure (Celery, Redis, etc.).

    Implémentations:
        - Bus inline (par défaut): exécute directement les handlers via le registry.
        - `CeleryEventBus`: envoie l'event vers un worker Celery (asynchrone).
    """

    @abstractmethod
    def publish(self, event: DomainEvent) -> None:
        raise NotImplementedError

    def publish_many(self, events: list[DomainEvent]) -> None:
        for e in events:
            self.publish(e)


_BUS_HOLDER: dict[str, EventBus | None] = {"bus": None}


def set_event_bus(bus: EventBus) -> None:
    """Configure le bus global (wiring application).

    Args:
        bus: Instance de bus à utiliser par défaut.
    """
    _BUS_HOLDER["bus"] = bus


def get_event_bus() -> EventBus:
    """Retourne le bus par défaut.

    Par défaut: bus synchrone (inline) pour ne pas dépendre d'un worker Celery.
    En prod, on pourra le remplacer via `set_event_bus(CeleryEventBus())`.
    """
    bus = _BUS_HOLDER["bus"]
    if bus is None:
        # ⚠️ Clean Architecture: la couche Application ne doit pas dépendre
        # de l'infrastructure (Celery/Redis) ni du legacy (`services/*`).
        #
        # Donc, par défaut, on utilise un bus "no-op". Le wiring (ex: app.py)
        # doit appeler `set_event_bus(...)` avec une implémentation Infrastructure
        # (`InMemoryEventBus` en dev/tests, `CeleryEventBus` en prod).
        bus = _NoopEventBus()
        _BUS_HOLDER["bus"] = bus
    return bus


def publish_event(event: DomainEvent) -> None:
    """Publie un événement via le bus configuré."""
    get_event_bus().publish(event)


class _NoopEventBus(EventBus):
    """Bus par défaut (no-op).

    Motivation:
        - Empêcher toute dépendance Application -> services/infrastructure.
        - Forcer un wiring explicite du bus d'events.
    """

    @override
    def publish(self, event: DomainEvent) -> None:
        # Intentionnel: on ne dispatch pas sans wiring explicite.
        # Les use-cases restent testables (les tests peuvent injecter un bus fake).
        return None
