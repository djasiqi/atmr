"""Producer Kafka pour l'admission tracking async.

Fallback comportement:
  - Si Kafka est disponible  → enqueue async, retourne queued=True
  - Si Kafka est indisponible → retourne queued=False, l'appelant
    (routes/driver.py) bascule sur l'insertion synchrone en DB.

Initialisation lazy par défaut (pas de connexion broker à l’import) :
Alembic, healthcheck et démarrage Gunicorn restent fonctionnels même si
Kafka est absent (variable TRACKING_INGEST_EAGER_INIT=true pour l’ancien
comportement).

Aucune exception ne doit se propager hors de ce module : toutes les
erreurs Kafka sont absorbées et tracées via le logger, ce qui garantit
qu'un broker KO ne casse pas l'endpoint HTTP chauffeur.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any

from services.region_router import kafka_partition_key

from .kafka_topics import TOPIC_DRIVER_LOCATION_RAW

logger = logging.getLogger(__name__)

KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
TRACKING_INGEST_ASYNC_ENABLED = (
    os.getenv("TRACKING_INGEST_ASYNC_ENABLED", "false").lower() == "true"
)
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
)
KAFKA_ACKS = os.getenv("KAFKA_ACKS", "all")
KAFKA_LINGER_MS = int(os.getenv("KAFKA_LINGER_MS", "5"))
KAFKA_BATCH_SIZE = int(os.getenv("KAFKA_BATCH_SIZE", "32768"))
KAFKA_COMPRESSION_TYPE = os.getenv("KAFKA_COMPRESSION_TYPE", "gzip")
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_SASL_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "")
KAFKA_SASL_USERNAME = os.getenv("KAFKA_SASL_USERNAME", "")
KAFKA_SASL_PASSWORD = os.getenv("KAFKA_SASL_PASSWORD", "")
KAFKA_SSL_CAFILE = os.getenv("KAFKA_SSL_CAFILE", "")
KAFKA_SSL_CERTFILE = os.getenv("KAFKA_SSL_CERTFILE", "")
KAFKA_SSL_KEYFILE = os.getenv("KAFKA_SSL_KEYFILE", "")

# Délai max pour attendre l'ACK broker avant de déclarer Kafka indisponible
# et basculer sur le chemin synchrone. Valeur conservatrice : assez longue
# pour absorber un leader election (< 1s normal), assez courte pour ne pas
# bloquer l'endpoint HTTP chauffeur en production.
KAFKA_PRODUCE_TIMEOUT_S = float(os.getenv("KAFKA_PRODUCE_TIMEOUT_S", "0.5"))

# Initialisation lazy : `flask db upgrade`, healthcheck et import de l’app ne tentent
# plus de joindre Kafka tant qu’aucune position n’est mise en file.
TRACKING_INGEST_EAGER_INIT = (
    os.getenv("TRACKING_INGEST_EAGER_INIT", "false").lower() == "true"
)
KAFKA_PRODUCER_INIT_BACKOFF_S = float(os.getenv("KAFKA_PRODUCER_INIT_BACKOFF_S", "30"))


def _kafka_security_config() -> dict[str, Any]:
    cfg: dict[str, Any] = {"security_protocol": KAFKA_SECURITY_PROTOCOL}
    if KAFKA_SASL_MECHANISM:
        cfg["sasl_mechanism"] = KAFKA_SASL_MECHANISM
    if KAFKA_SASL_USERNAME:
        cfg["sasl_plain_username"] = KAFKA_SASL_USERNAME
    if KAFKA_SASL_PASSWORD:
        cfg["sasl_plain_password"] = KAFKA_SASL_PASSWORD
    if KAFKA_SSL_CAFILE:
        cfg["ssl_cafile"] = KAFKA_SSL_CAFILE
    if KAFKA_SSL_CERTFILE:
        cfg["ssl_certfile"] = KAFKA_SSL_CERTFILE
    if KAFKA_SSL_KEYFILE:
        cfg["ssl_keyfile"] = KAFKA_SSL_KEYFILE
    return cfg


class TrackingIngestProducer:
    def __init__(self) -> None:
        super().__init__()
        self._producer = None
        self._initialized = False
        self._next_init_attempt_mono = 0.0
        if (
            TRACKING_INGEST_EAGER_INIT
            and KAFKA_ENABLED
            and TRACKING_INGEST_ASYNC_ENABLED
        ):
            self._init_producer(reset_backoff=True)

    def _maybe_init_producer(self, *, reset_backoff: bool = False) -> None:
        if not KAFKA_ENABLED or not TRACKING_INGEST_ASYNC_ENABLED:
            return
        if self._initialized and self._producer is not None:
            return
        now = time.monotonic()
        if reset_backoff:
            self._next_init_attempt_mono = 0.0
        elif now < self._next_init_attempt_mono:
            return
        self._init_producer(reset_backoff=reset_backoff)

    def _init_producer(self, *, reset_backoff: bool = False) -> None:
        if reset_backoff:
            self._next_init_attempt_mono = 0.0
        self._producer = None
        self._initialized = False
        try:
            from kafka import KafkaProducer as KP

            self._producer = KP(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                acks=KAFKA_ACKS,
                linger_ms=KAFKA_LINGER_MS,
                batch_size=KAFKA_BATCH_SIZE,
                compression_type=KAFKA_COMPRESSION_TYPE,
                enable_idempotence=True,
                retries=3,
                **_kafka_security_config(),
            )
            self._initialized = True
            self._next_init_attempt_mono = 0.0
            logger.info("[tracking_ingest] Kafka producer initialized")
        except ImportError:
            logger.error(
                "[tracking_ingest] kafka-python missing, install dependency to enable async tracking ingest"
            )
            self._next_init_attempt_mono = time.monotonic() + max(
                KAFKA_PRODUCER_INIT_BACKOFF_S, 60.0
            )
        except Exception as exc:
            # Ne jamais faire échouer l’import Flask / Alembic / Gunicorn pour un broker KO
            logger.warning(
                "[tracking_ingest] Kafka indisponible — ingest async désactivé (%s: %s)",
                type(exc).__name__,
                exc,
            )
            self._next_init_attempt_mono = (
                time.monotonic() + KAFKA_PRODUCER_INIT_BACKOFF_S
            )

    @property
    def enabled(self) -> bool:
        return self._initialized

    def enqueue(
        self,
        *,
        driver_id: int,
        payload: dict[str, Any],
        source: str = "http",
        company_id: int | None = None,
    ) -> dict[str, Any]:
        """Tente d'envoyer un point GPS vers Kafka.

        Retourne toujours un dict — ne lève jamais d'exception.
        Le champ ``queued`` indique si l'envoi a réussi :
          - True  → point accepté par le broker, l'appelant peut retourner 202
          - False → Kafka indisponible ou désactivé, l'appelant doit basculer
                    sur l'insertion synchrone en base de données
        """
        trace_id = str(uuid.uuid4())
        now_ms = int(time.time() * 1000)
        message = {
            "trace_id": trace_id,
            "driver_id": driver_id,
            "source": source,
            "received_at_ms": now_ms,
            "payload": payload,
        }

        if not self._initialized or self._producer is None:
            return {"queued": False, "trace_id": trace_id, "reason": "kafka_disabled"}

        payload_region_id_obj = payload.get("region_id")
        payload_company_id_obj = payload.get("company_id")
        region_id = (
            payload_region_id_obj if isinstance(payload_region_id_obj, str) else None
        )
        partition_company_id = (
            int(company_id)
            if company_id is not None
            else (
                payload_company_id_obj
                if isinstance(payload_company_id_obj, int)
                else None
            )
        )
        key = kafka_partition_key(
            region_id=region_id,
            company_id=partition_company_id,
            driver_id=driver_id,
        )

        try:
            future = self._producer.send(
                TOPIC_DRIVER_LOCATION_RAW, value=message, key=key
            )
            future.get(timeout=KAFKA_PRODUCE_TIMEOUT_S)
            try:
                from services.monitoring.driver_location_metrics import (
                    inc_tracking_kafka_messages_produced,
                )

                inc_tracking_kafka_messages_produced(topic=TOPIC_DRIVER_LOCATION_RAW)
            except Exception:
                logger.debug(
                    "[tracking_ingest] produced metric unavailable", exc_info=True
                )
            return {
                "queued": True,
                "trace_id": trace_id,
                "topic": TOPIC_DRIVER_LOCATION_RAW,
            }
        except Exception as exc:
            # Absorbe toutes les erreurs Kafka (NoBrokersAvailable, KafkaTimeoutError,
            # NotLeaderForPartitionError, etc.) et signale à l'appelant de basculer
            # sur le chemin synchrone. Un log WARN (pas ERROR) car c'est un fallback
            # attendu en cas de maintenance broker ou réseau dégradé.
            logger.warning(
                "[tracking_ingest] Kafka enqueue failed, fallback to sync path (driver_id=%s trace_id=%s): %s",
                driver_id,
                trace_id,
                exc,
            )
            try:
                from services.monitoring.driver_location_metrics import (
                    inc_tracking_kafka_publish_errors,
                )

                inc_tracking_kafka_publish_errors(
                    topic=TOPIC_DRIVER_LOCATION_RAW, stage="raw_enqueue_failed"
                )
            except Exception:
                logger.debug(
                    "[tracking_ingest] publish error metric unavailable", exc_info=True
                )
            # Tenter de réinitialiser le producer pour le prochain appel.
            try:
                self._initialized = False
                self._producer = None
                self._maybe_init_producer(reset_backoff=True)
            except Exception:
                pass
            return {"queued": False, "trace_id": trace_id, "reason": "kafka_error"}


tracking_ingest_producer = TrackingIngestProducer()


def enqueue_tracking_event(
    *,
    driver_id: int,
    payload: dict[str, Any],
    source: str = "http",
    company_id: int | None = None,
) -> dict[str, Any]:
    return tracking_ingest_producer.enqueue(
        driver_id=driver_id,
        payload=payload,
        source=source,
        company_id=company_id,
    )
