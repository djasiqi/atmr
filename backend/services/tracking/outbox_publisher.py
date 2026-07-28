"""Publisher outbox GPS — Annexe A.6 (pg_try_advisory_lock 2-int).

Publication ordonnée par (session_generation, sequence_id) par chauffeur.
Lock session tenu sur UNE seule connexion pendant Kafka (TX SQL fermée).
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

# Namespace advisory lock 2-int (évite hashtext collisions)
OUTBOX_LOCK_NAMESPACE = int(os.getenv("TRACKING_OUTBOX_LOCK_NAMESPACE", "42001"))


def _database_url() -> str:
    # Advisory locks session-level : éviter PgBouncer en mode transaction.
    for key in (
        "DATABASE_URL_DIRECT",
        "SQLALCHEMY_DATABASE_URI_DIRECT",
        "POSTGRES_URL",
    ):
        url = os.getenv(key)
        if url:
            break
    else:
        url = os.getenv("DATABASE_URL") or os.getenv("SQLALCHEMY_DATABASE_URI")
    if not url:
        # Fallback compose Kafka : POSTGRES_* (souvent via pgbouncer) → forcer postgres:5432
        from urllib.parse import quote_plus

        user = os.getenv("POSTGRES_USER", "atmr")
        password = os.getenv("POSTGRES_PASSWORD", "")
        db = os.getenv("POSTGRES_DB", "atmr")
        if not password:
            raise RuntimeError("DATABASE_URL/POSTGRES_* manquant pour outbox_publisher")
        url = (
            f"postgresql://{quote_plus(user)}:{quote_plus(password)}"
            f"@postgres:5432/{db}?sslmode=disable"
        )
    url = url.replace("postgres://", "postgresql://", 1)
    if "@pgbouncer:" in url or "@atmr-pgbouncer" in url:
        url = url.replace("@pgbouncer:", "@postgres:").replace(
            "@atmr-pgbouncer:", "@postgres:"
        )
        url = url.replace(":6432/", ":5432/")
    return url


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
            drivers = (
                conn.execute(
                    text(
                        """
                    SELECT DISTINCT driver_id
                    FROM tracking_event_outbox
                    WHERE published_at IS NULL
                    ORDER BY driver_id
                    LIMIT 50
                    """
                    )
                )
                .scalars()
                .all()
            )

        for driver_id in drivers:
            published += self._publish_for_driver(int(driver_id))
        return published

    def _publish_for_driver(self, driver_id: int) -> int:
        # Même lock_conn pour acquire → SELECT → Kafka → UPDATE → unlock.
        # Commit après lock : TX SQL fermée pendant l'appel réseau Kafka.
        with self._engine.connect() as lock_conn:
            got = lock_conn.execute(
                text("SELECT pg_try_advisory_lock(:ns, :driver_id)"),
                {"ns": OUTBOX_LOCK_NAMESPACE, "driver_id": driver_id},
            ).scalar()
            if not got:
                return 0
            lock_conn.commit()

            try:
                rows = (
                    lock_conn.execute(
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
                    )
                    .mappings()
                    .all()
                )
                lock_conn.commit()

                if not rows:
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
                        lock_conn.execute(
                            text(
                                """
                                UPDATE tracking_event_outbox
                                SET published_at = :ts, attempts = attempts + 1
                                WHERE id = :id
                                """
                            ),
                            {"ts": datetime.now(UTC), "id": int(row["id"])},
                        )
                        lock_conn.commit()
                        count += 1
                except Exception:
                    logger.exception(
                        "[outbox] publish failed driver_id=%s — republication at-least-once",
                        driver_id,
                    )
                    err_id = (
                        int(rows[count]["id"])
                        if count < len(rows)
                        else int(rows[-1]["id"])
                    )
                    lock_conn.execute(
                        text(
                            """
                            UPDATE tracking_event_outbox
                            SET attempts = attempts + 1, last_error = :err
                            WHERE id = :id
                            """
                        ),
                        {"err": "kafka_publish_failed", "id": err_id},
                    )
                    lock_conn.commit()
                return count
            finally:
                lock_conn.execute(
                    text("SELECT pg_advisory_unlock(:ns, :driver_id)"),
                    {"ns": OUTBOX_LOCK_NAMESPACE, "driver_id": driver_id},
                )
                lock_conn.commit()

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
