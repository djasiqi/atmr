from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class DomainEvent:
    """Base pour les événements de domaine."""

    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    correlation_id: str | None = None

    # Nom stable pour la sérialisation/dispatch
    event_type: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type
        # datetime -> iso
        data["occurred_at"] = self.occurred_at.isoformat()
        return data
