"""Écriture canonique et idempotente de ``Booking.arrived_at``.

Tous les chemins (jalon chauffeur, PATCH dispatcher) doivent passer ici.
Ne commit pas : l'appelant garde sa transaction.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


def record_booking_arrival(booking: Any, *, now: datetime) -> bool:
    """Pose ``arrived_at`` une seule fois.

    Returns:
        True si la valeur a été écrite, False si elle existait déjà.
    """
    if getattr(booking, "arrived_at", None) is not None:
        return False
    booking.arrived_at = now
    return True
