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

# Contrat Kafka v3 (RF=3 / minISR=2) — suffixe = version contrat
TOPIC_DRIVER_LOCATION_RAW_V3 = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_RAW_V3", "driver.location.raw.v3"
)
TOPIC_DRIVER_LOCATION_PROCESSED_V3 = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED_V3", "driver.location.processed.v3"
)
TOPIC_DRIVER_LOCATION_ENRICHED_V3 = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_ENRICHED_V3", "driver.location.enriched.v3"
)
TOPIC_DRIVER_LOCATION_DLQ_V3 = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_DLQ_V3", "driver.location.dlq.v3"
)
TOPIC_DRIVER_LOCATION_RAW_SHADOW_V3 = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_RAW_SHADOW_V3", "driver.location.raw.shadow.v3"
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
