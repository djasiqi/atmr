"""Contrat GPS partagé backend / ws-service (plan Kafka-first v5)."""

from .envelope import TrackingEnvelope, normalize_location_event_id
from .session import SESSION_STATUSES, first_sequence_id
from .schema_version import PAYLOAD_SCHEMA_VERSION, KAFKA_CONTRACT_VERSION

__all__ = [
    "TrackingEnvelope",
    "normalize_location_event_id",
    "SESSION_STATUSES",
    "first_sequence_id",
    "PAYLOAD_SCHEMA_VERSION",
    "KAFKA_CONTRACT_VERSION",
]
