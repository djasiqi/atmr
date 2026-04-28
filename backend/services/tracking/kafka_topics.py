"""Topics Kafka utilises par le pipeline tracking et events Phase 3."""

from __future__ import annotations

import os

TOPIC_DRIVER_LOCATION_RAW = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_RAW", "driver.location.raw"
)
TOPIC_DRIVER_LOCATION_PROCESSED = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED", "driver.location.processed"
)
TOPIC_DRIVER_LOCATION_ENRICHED = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_ENRICHED", "driver.location.enriched"
)
TOPIC_DRIVER_LOCATION_REALTIME = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_REALTIME", "driver.location.realtime"
)
TOPIC_DRIVER_LOCATION_DLQ = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_DLQ", "driver.location.dlq"
)

TOPIC_MISSION_EVENTS = os.getenv("KAFKA_TOPIC_MISSION_EVENTS", "mission.events")
TOPIC_NOTIFICATION_EVENTS = os.getenv(
    "KAFKA_TOPIC_NOTIFICATION_EVENTS", "notification.events"
)
TOPIC_DISPATCH_EVENTS = os.getenv("KAFKA_TOPIC_DISPATCH_EVENTS", "dispatch.events")

# Alias legacy pour compatibilite backward pendant le dual-run.
TOPIC_DRIVER_LOCATION_VALIDATED = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_VALIDATED", TOPIC_DRIVER_LOCATION_PROCESSED
)

