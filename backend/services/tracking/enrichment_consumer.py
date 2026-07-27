"""Consumer processed → enrichissement OSRM → topic enriched.v3 (Annexe A.5).

N'écrit JAMAIS Redis directement — ws-service est le seul writer Redis.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

TOPIC_PROCESSED = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED_V3",
    os.getenv("KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED", "driver.location.processed"),
)
TOPIC_ENRICHED_V3 = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_ENRICHED_V3",
    "driver.location.enriched.v3",
)
GROUP_ID = os.getenv(
    "KAFKA_TRACKING_ENRICHMENT_GROUP",
    "gps-enrichment-v3",
)
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
)
ENRICHMENT_VERSION = int(os.getenv("TRACKING_ENRICHMENT_VERSION", "1"))
KAFKA_PUBLISH_ACK_TIMEOUT_S = float(os.getenv("KAFKA_PUBLISH_ACK_TIMEOUT_S", "2.0"))


def _database_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI")
    if not url:
        raise RuntimeError("DATABASE_URL manquant pour enrichment_consumer")
    return url.replace("postgres://", "postgresql://", 1)


def _snap_coords(lat: float, lon: float) -> tuple[float, float, str]:
    """Snap OSRM best-effort ; fallback = coords brutes."""
    try:
        from services.geolocation.osrm_client import snap_to_road  # type: ignore

        snapped = snap_to_road(lat, lon)
        if snapped and "latitude" in snapped and "longitude" in snapped:
            return float(snapped["latitude"]), float(snapped["longitude"]), "osrm"
    except Exception:
        logger.debug("[enrichment] OSRM snap unavailable", exc_info=True)
    return lat, lon, "raw_fallback"


def persist_enrichment(
    engine: Engine,
    *,
    driver_id: int,
    location_event_id: str,
    canonical_lat: float,
    canonical_lon: float,
    canonical_source: str,
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO driver_location_enrichments (
                    driver_id, location_event_id, enrichment_version,
                    canonical_latitude, canonical_longitude,
                    canonical_source, processing_status, enriched_at
                ) VALUES (
                    :driver_id, :eid, :ver,
                    :lat, :lon, :src, 'done', :ts
                )
                ON CONFLICT (driver_id, location_event_id, enrichment_version)
                DO NOTHING
                """
            ),
            {
                "driver_id": driver_id,
                "eid": location_event_id,
                "ver": ENRICHMENT_VERSION,
                "lat": canonical_lat,
                "lon": canonical_lon,
                "src": canonical_source,
                "ts": datetime.now(UTC),
            },
        )


class TrackingEnrichmentConsumer:
    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine or create_engine(_database_url(), pool_pre_ping=True)
        self._consumer = None
        self._producer = None
        self._running = False
        self._initialized = False

    def initialize(self) -> bool:
        try:
            from kafka import KafkaConsumer, KafkaProducer
        except Exception:
            logger.exception("[enrichment] kafka-python unavailable")
            return False
        self._consumer = KafkaConsumer(
            TOPIC_PROCESSED,
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
            group_id=GROUP_ID,
            enable_auto_commit=False,
            auto_offset_reset="earliest",
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        )
        self._producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
            acks="all",
            enable_idempotence=True,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
        )
        self._initialized = True
        return True

    def _commit_record(self, record: Any) -> None:
        from kafka.structs import OffsetAndMetadata, TopicPartition

        assert self._consumer is not None
        tp = TopicPartition(record.topic, record.partition)
        self._consumer.commit({tp: OffsetAndMetadata(record.offset + 1, "", -1)})

    def _process(self, message: dict[str, Any]) -> None:
        payload = message.get("payload")
        if not isinstance(payload, dict):
            payload = message
        driver_id = int(message.get("driver_id") or payload.get("driver_id"))
        eid = str(
            message.get("location_event_id") or payload.get("location_event_id") or ""
        )
        if not eid:
            return
        lat = float(payload.get("latitude", payload.get("lat")))
        lon = float(payload.get("longitude", payload.get("lon")))
        canon_lat, canon_lon, src = _snap_coords(lat, lon)
        persist_enrichment(
            self._engine,
            driver_id=driver_id,
            location_event_id=eid,
            canonical_lat=canon_lat,
            canonical_lon=canon_lon,
            canonical_source=src,
        )
        enriched_msg = {
            "type": "driver.location.enriched",
            "driver_id": driver_id,
            "company_id": message.get("company_id") or payload.get("company_id"),
            "location_event_id": eid,
            "session_generation": message.get("session_generation")
            or payload.get("session_generation"),
            "sequence_id": message.get("sequence_id") or payload.get("sequence_id"),
            "payload": {
                **payload,
                "location_event_id": eid,
                "canonical_latitude": canon_lat,
                "canonical_longitude": canon_lon,
                "canonical_source": src,
                "enrichment_version": ENRICHMENT_VERSION,
            },
        }
        assert self._producer is not None
        future = self._producer.send(
            TOPIC_ENRICHED_V3,
            key=f"driver_{driver_id}",
            value=enriched_msg,
        )
        future.get(timeout=KAFKA_PUBLISH_ACK_TIMEOUT_S)

    def start(self) -> None:
        if not self._initialized and not self.initialize():
            raise RuntimeError("enrichment consumer init failed")
        assert self._consumer is not None
        self._running = True

        def _stop(*_args: Any) -> None:
            self._running = False

        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        logger.info(
            "[enrichment] start processed=%s enriched=%s",
            TOPIC_PROCESSED,
            TOPIC_ENRICHED_V3,
        )
        while self._running:
            polled = self._consumer.poll(timeout_ms=1000)
            for _tp, records in polled.items():
                for record in records:
                    try:
                        value = record.value if isinstance(record.value, dict) else {}
                        self._process(value)
                        self._commit_record(record)
                    except Exception:
                        logger.exception(
                            "[enrichment] process failed — no commit offset=%s",
                            getattr(record, "offset", None),
                        )
                        time.sleep(1.0)
                        raise
            time.sleep(0.01)


def run_tracking_enrichment_consumer() -> None:
    TrackingEnrichmentConsumer().start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_tracking_enrichment_consumer()
