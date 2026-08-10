"""Alias de compatibilité vers ``services.notifications`` (core + push).

Les modules historiques importent encore ``services.notification_service`` ;
ce shim évite de casser les tests et le code legacy après la consolidation B2.
"""

from __future__ import annotations

from services.notifications.core import (
    notify_booking_assigned,
    notify_booking_cancelled,
    notify_booking_update,
    notify_dispatch_run_completed,
    notify_driver_new_booking,
)
from services.notifications.push import send_push_message

__all__ = [
    "notify_booking_assigned",
    "notify_booking_cancelled",
    "notify_booking_update",
    "notify_dispatch_run_completed",
    "notify_driver_new_booking",
    "send_push_message",
]
