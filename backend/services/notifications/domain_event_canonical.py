# backend/services/notifications/domain_event_canonical.py
"""Payload canonique pour toutes les notifications (Socket.IO + Push).

Toutes les notifications doivent partir d'un DomainEvent canonical pour garantir
cohérence, dédup et routage (exclude_actor, policy, throttle).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# Types littéraux pour actor_role, severity, category
ACTOR_ROLE_DRIVER = "driver"
ACTOR_ROLE_COMPANY = "company"
ACTOR_ROLE_SYSTEM = "system"

SEVERITY_INFO = "info"
SEVERITY_IMPORTANT = "important"
SEVERITY_CRITICAL = "critical"

CATEGORY_BOOKING = "booking"
CATEGORY_DRIVER_STATUS = "driver_status"
CATEGORY_DELAY = "delay"
CATEGORY_ADMIN = "admin"


@dataclass(frozen=False)
class DomainEventCanonical:
    """Event canonique pour le routage des notifications.

    Utilisé par NotificationRouter pour décider socket_emits[] et push_requests[].
    """

    type: str  # ex: BOOKING_ASSIGNED, DRIVER_EN_ROUTE, BOOKING_UPDATED
    booking_id: int | None = None
    company_id: int | None = None
    driver_id: int | None = None
    actor_role: str | None = None  # driver | company | system
    actor_id: int | None = None
    severity: str = SEVERITY_INFO  # info | important | critical
    category: str = CATEGORY_BOOKING  # booking | driver_status | delay | admin
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))
    title: str = ""
    body: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    delta: dict[str, Any] | None = None  # pour BOOKING_UPDATED "significant change"

    def dedupe_key(self, version_or_hash: str = "v1") -> str:
        """Clé stable pour dédup: booking:{id}:type:{type}:v{version}."""
        bid = self.booking_id or 0
        return f"booking:{bid}:type:{self.type}:{version_or_hash}"

    def collapse_key(self) -> str:
        """Clé pour regroupement FCM/APNs: booking:{id}:{category}."""
        bid = self.booking_id or 0
        return f"booking:{bid}:{self.category}"

    def to_payload(self) -> dict[str, Any]:
        """Payload pour socket et push (data)."""
        payload: dict[str, Any] = {
            "type": self.type,
            "booking_id": self.booking_id,
            "company_id": self.company_id,
            "driver_id": self.driver_id,
            "actor_role": self.actor_role,
            "actor_id": self.actor_id,
            "severity": self.severity,
            "category": self.category,
            "dedupe_key": self.dedupe_key(),
            "ts": self.ts.isoformat(),
            "title": self.title,
            "body": self.body,
            "data": {**self.data, "delta": self.delta or {}},
        }
        return payload


def build_dedupe_version(changes: dict[str, Any] | None) -> str:
    """Version courte pour dedupe_key à partir des changements (BOOKING_UPDATED)."""
    if not changes:
        return "v1"
    h = hashlib.sha256(str(sorted(changes.items())).encode()).hexdigest()[:8]
    return f"v{h}"
