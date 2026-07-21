"""NegotiationPolicy + ResponsePolicy — V1.2 / V1.3 (defaults code)."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NegotiationPolicy:
    """Règles de négociation par défaut (configurables plus tard en DB)."""

    counter_enabled: bool = False
    max_rounds: int = 5
    allowed_fields_by_action: dict[str, frozenset[str]] | None = None

    def allows_counter(self) -> bool:
        if not self.counter_enabled:
            return False
        return os.getenv("TRANSPORT_ACTION_COUNTER_ENABLED", "false").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def allowed_counter_fields(self, action_type: str) -> frozenset[str]:
        defaults = {
            "CHANGE_TIME": frozenset({"scheduled_time"}),
            "CHANGE_DATE": frozenset({"scheduled_time"}),
            "CHANGE_PICKUP_ADDRESS": frozenset(
                {"pickup_location", "pickup_lat", "pickup_lon"}
            ),
            "CHANGE_DROPOFF_ADDRESS": frozenset(
                {"dropoff_location", "dropoff_lat", "dropoff_lon"}
            ),
            "CANCELLATION": frozenset(),  # commercial_terms seulement
            "CHANGE_OTHER": frozenset({"scheduled_time", "notes_medical"}),
        }
        mapping = self.allowed_fields_by_action or defaults
        return mapping.get(action_type, frozenset())


@dataclass(frozen=True, slots=True)
class ResponsePolicy:
    """Délais de relance / expiration — defaults opérationnels (annexe domain)."""

    ttl_minutes_default: int = 120
    remind_after_minutes_gt_24h: int = 240
    remind_after_minutes_2_24h: int = 30
    remind_after_minutes_lt_2h: int = 15
    expire_after_hours_gt_24h: int = 12

    def remind_interval_minutes(self, minutes_to_departure: int | None) -> int:
        if minutes_to_departure is None:
            return self.remind_after_minutes_2_24h
        if minutes_to_departure > 24 * 60:
            return self.remind_after_minutes_gt_24h
        if minutes_to_departure > 120:
            return self.remind_after_minutes_2_24h
        return self.remind_after_minutes_lt_2h


def default_negotiation_policy() -> NegotiationPolicy:
    return NegotiationPolicy(
        counter_enabled=os.getenv("TRANSPORT_ACTION_COUNTER_ENABLED", "false").lower()
        in ("1", "true", "yes", "on")
    )


def default_response_policy() -> ResponsePolicy:
    raw = os.getenv("INSTITUTION_CHANGE_REQUEST_TTL_MINUTES", "120")
    try:
        ttl = max(5, int(raw))
    except ValueError:
        ttl = 120
    return ResponsePolicy(ttl_minutes_default=ttl)
