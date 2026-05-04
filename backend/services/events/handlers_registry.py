"""Compat: registry historique des handlers d'événements.

Certaines parties de l'infra (ex: `InMemoryEventBus`) importent encore
`services.events.handlers_registry`. Le module officiel est `services.events.registry`.
On expose donc les mêmes symboles ici.
"""

from __future__ import annotations

from services.events.registry import dispatch_event, register

__all__ = ["dispatch_event", "register"]
