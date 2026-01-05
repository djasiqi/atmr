from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DispatchRunRequestCommand:
    """Commande (input) pour le dispatch."""

    company_id: int
    body: dict[str, Any]
