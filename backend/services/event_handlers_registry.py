"""Compatibilité: `from services import event_handlers_registry` (tests + bus mémoire)."""

from __future__ import annotations

from services.events.registry import _HANDLERS, dispatch_event, register

__all__ = ["_HANDLERS", "dispatch_event", "register"]
