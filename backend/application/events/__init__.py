"""Bus d'événements (Application).

Expose une API stable pour publier des Domain Events sans dépendre de l'infra.
"""

from .event_bus import EventBus, get_event_bus, publish_event, set_event_bus

__all__ = [
    "EventBus",
    "get_event_bus",
    "publish_event",
    "set_event_bus",
]
