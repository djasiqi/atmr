"""Publisher outbox GPS — Annexe A.6 (pg_try_advisory_lock).

Publication ordonnée par (session_generation, sequence_id) par chauffeur.
Ne garde PAS de transaction SQL ouverte pendant l'appel Kafka.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

TOPIC_PROCESSED = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED",
    "driver.location.processed",
)
# Phase 2/3 : bascule possible vers driver.location.processed.v3
TOPIC_PROCESSED_V3 = os.getenv(
    "KAFKA_TOPIC_DRIVER_LOCATION_PROCESSED_V3",
    "driver.location.processed.v3",
)
USE_V3_TOPIC = os.getenv("TRACKING_OUTBOX_USE_V3_TOPIC", "false").lower() == "true"
OUTBOX_BATCH_PER_DRIVER = int(os.getenv("TRACKING_OUTBOX_BATCH_PER_DRIVER", "20"))
OUTBOX_POLL_INTERVAL_S = float(os.getenv("TRACKING_OUTBOX_POLL_INTERVAL_S", "0.5"))
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
)
KAFKA_PUBLISH_ACK_TIMEOUT_S = float(os.getenv("KAFKA_PUBLISH_ACK_TIMEOUT_S", "2.0"))


def _database_url() -> str:
    url = os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI")
    if not url:
        raise RuntimeError("DATABASE_URL manquant pour outbox_publisher")
    return url.replace("postgres://", "postgresql://", 1)


class TrackingOutboxPublisher:
    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine or create_engine(_database_url(), pool_pre_ping=True)
        self._producer = None
        self._running = False

    def _ensure_producer(self) -> Any:
        if self._producer is not None:
            return self._producer
        from kafka import KafkaProducer

        self._producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
            acks="all",
            enable_idempotence=True,
            value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
            key_serializer=lambda k: str(k).encode("utf-8"),
        )
        return self._producer

    def _target_topic(self) -> str:
        return TOPIC_PROCESSED_V3 if USE_V3_TOPIC else TOPIC_PROCESSED

    def publish_once(self) -> int:
        """Traite un round : drivers distincts avec pending, lock advisory, publish."""
        published = 0
        with self._engine.connect() as conn:
            drivers = conn.execute(
                text(
                    """
                    SELECT DISTINCT driver_id
                    FROM tracking_event_outbox
                    WHERE published_at IS NULL
                    ORDER BY driver_id
                    LIMIT 50
                    """
                )
            ).scalars().all()

        for driver_id in drivers:
            published += self._publish_for_driver(int(driver_id))
        return published

    def _publish_for_driver(self, driver_id: int) -> int:
        # Annexe A.6 : pg_try_advisory_lock(hashtext(...)) sur connexion dédiée
        # — jamais xact_lock pendant l'appel réseau Kafka.
        with self._engine.connect() as conn:
            got = conn.execute(
                text(
                    "SELECT pg_try_advisory_lock("
                    "hashtext('tracking_outbox:' || CAST(:driver_id AS text)))"
                ),
                {"driver_id": driver_id},
            ).scalar()
            if not got:
                return 0

            try:
                rows = conn.execute(
                    text(
                        """
                        SELECT id, event_id, location_event_id, payload,
                               session_generation, sequence_id
                        FROM tracking_event_outbox
                        WHERE driver_id = :driver_id AND published_at IS NULL
                        ORDER BY session_generation ASC, sequence_id ASC
                        LIMIT :lim
                        """
                    ),
                    {"driver_id": driver_id, "lim": OUTBOX_BATCH_PER_DRIVER},
                ).mappings().all()
                conn.commit()
            except Exception:
                conn.execute(
                    text(
                        "SELECT pg_advisory_unlock("
                        "hashtext('tracking_outbox:' || CAST(:driver_id AS text)))"
                    ),
                    {"driver_id": driver_id},
                )
                conn.commit()
                raise

        if not rows:
            with self._engine.connect() as conn:
                conn.execute(
                    text(
                        "SELECT pg_advisory_unlock("
                        "hashtext('tracking_outbox:' || CAST(:driver_id AS text)))"
                    ),
                    {"driver_id": driver_id},
                )
                conn.commit()
            return 0

        producer = self._ensure_producer()
        topic = self._target_topic()
        count = 0
        try:
            for row in rows:
                payload = row["payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                future = producer.send(
                    topic,
                    key=f"driver_{driver_id}",
                    value=payload,
                )
                future.get(timeout=KAFKA_PUBLISH_ACK_TIMEOUT_S)
                with self._engine.connect() as conn:
                    conn.execute(
                        text(
                            """
                            UPDATE tracking_event_outbox
                            SET published_at = :ts, attempts = attempts + 1
                            WHERE id = :id
                            """
                        ),
                        {"ts": datetime.now(UTC), "id": int(row["id"])},
                    )
                    conn.commit()
                count += 1
        except Exception:
            logger.exception(
                "[outbox] publish failed driver_id=%s — republication at-least-once",
                driver_id,
            )
            with self._engine.connect() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE tracking_event_outbox
                        SET attempts = attempts + 1, last_error = :err
                        WHERE id = :id
                        """
                    ),
                    {
                        "err": "kafka_publish_failed",
                        "id": int(rows[count]["id"])
                        if count < len(rows)
                        else int(rows[-1]["id"]),
                    },
                )
                conn.commit()
        finally:
            with self._engine.connect() as conn:
                conn.execute(
                    text(
                        "SELECT pg_advisory_unlock("
                        "hashtext('tracking_outbox:' || CAST(:driver_id AS text)))"
                    ),
                    {"driver_id": driver_id},
                )
                conn.commit()
        return count

    def run_loop(self) -> None:
        self._running = True
        logger.info("[outbox] start loop topic=%s", self._target_topic())
        while self._running:
            try:
                n = self.publish_once()
                if n == 0:
                    time.sleep(OUTBOX_POLL_INTERVAL_S)
            except Exception:
                logger.exception("[outbox] loop error")
                time.sleep(OUTBOX_POLL_INTERVAL_S)

    def stop(self) -> None:
        self._running = False
        if self._producer is not None:
            self._producer.flush(timeout=3)
            self._producer.close()


def run_tracking_outbox_publisher() -> None:
    publisher = TrackingOutboxPublisher()
    publisher.run_loop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_tracking_outbox_publisher()
