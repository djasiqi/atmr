from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CreateBookingCommand:
    """Commande (input) pour le cas d'usage de création de réservation."""

    user_id: int
    client_id: int
    data: dict[str, Any]
