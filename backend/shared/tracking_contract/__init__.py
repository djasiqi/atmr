"""Contrat GPS partagé backend / ws-service (plan Kafka-first v5)."""

from .envelope import TrackingEnvelope, normalize_location_event_id
from .schema_version import KAFKA_CONTRACT_VERSION, PAYLOAD_SCHEMA_VERSION
from .session import SESSION_STATUSES, first_sequence_id

__all__ = [
    "KAFKA_CONTRACT_VERSION",
    "PAYLOAD_SCHEMA_VERSION",
    "SESSION_STATUSES",
    "TrackingEnvelope",
    "first_sequence_id",
    "normalize_location_event_id",
]
