"""Consumer Kafka `driver.location.processed` → fanout Socket.IO entreprise (Flask).

PR-4 : le chemin HTTP async (202 + Kafka) ne passait pas par `fanout_driver_location_update`.
Ce worker réapplique le **même contrat** que le portail entreprise :
  - événements `driver_location_update` (+ `driver_live_state_update` si canonique)
  - room `company_{id}` (voir `services.realtime.socketio.get_company_room`)

Prérequis : `REDIS_URL` aligné sur l'API Flask si `message_queue` Socket.IO est utilisé
(multi-workers). Sinon le emit reste local au processus Flask uniquement.

Activation : `TRACKING_PROCESSED_FANOUT_ENABLED=true`, `KAFKA_ENABLED=true`,
`TRACKING_INGEST_ASYNC_ENABLED=true`.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from datetime import UTC, datetime
from typing import Any

from .kafka_topics import TOPIC_DRIVER_LOCATION_PROCESSED

logger = logging.getLogger(__name__)

KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
TRACKING_INGEST_ASYNC_ENABLED = (
    os.getenv("TRACKING_INGEST_ASYNC_ENABLED", "false").lower() == "true"
)
TRACKING_PROCESSED_FANOUT_ENABLED = (
    os.getenv("TRACKING_PROCESSED_FANOUT_ENABLED", "false").lower() == "true"
)
KAFKA_BOOTSTRAP_SERVERS = os.getenv(
    "KAFKA_BOOTSTRAP_SERVERS",
    "kafka-broker-1:29092,kafka-broker-2:29092,kafka-broker-3:29092",
)
KAFKA_PROCESSED_FANOUT_GROUP = os.getenv(
    "KAFKA_PROCESSED_FANOUT_GROUP", "tracking-processed-fanout-group"
)
KAFKA_AUTO_OFFSET_RESET = os.getenv("KAFKA_PROCESSED_FANOUT_OFFSET_RESET", "latest")
KAFKA_SECURITY_PROTOCOL = os.getenv("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT")
KAFKA_SASL_MECHANISM = os.getenv("KAFKA_SASL_MECHANISM", "")
KAFKA_SASL_USERNAME = os.getenv("KAFKA_SASL_USERNAME", "")
KAFKA_SASL_PASSWORD = os.getenv("KAFKA_SASL_PASSWORD", "")
KAFKA_SSL_CAFILE = os.getenv("KAFKA_SSL_CAFILE", "")
KAFKA_SSL_CERTFILE = os.getenv("KAFKA_SSL_CERTFILE", "")
KAFKA_SSL_KEYFILE = os.getenv("KAFKA_SSL_KEYFILE", "")
TRACKING_KAFKA_LAG_METRIC_ENABLED = (
    os.getenv("TRACKING_KAFKA_LAG_METRIC_ENABLED", "true").lower() == "true"
)
TRACKING_KAFKA_LAG_METRIC_INTERVAL_S = float(
    os.getenv("TRACKING_KAFKA_LAG_METRIC_INTERVAL_S", "15")
)


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


def _iso_from_ms(ms: Any) -> str | None:
    if not isinstance(ms, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=UTC).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _fanout_processed_message(envelope: dict[str, Any]) -> None:
    # Option A (voie socket → Kafka) : les points `source="socket_batch"` ont déjà
    # été fannés en direct par le handler socket (voie live faible latence). Kafka
    # ne sert ici qu'à la durabilité/analytics → on NE re-fanne PAS (évite le
    # doublon d'événements `driver_location_update` côté entreprise). Ce skip est
    # AVANT les imports lourds (certains chargent des modules exigeant les clés de
    # chiffrement) pour ne rien déclencher inutilement sur la voie socket.
    if str(envelope.get("source") or "") == "socket_batch":
        return

    from celery_app import get_flask_app
    from ext import db
    from models import Driver
    from services.company_driver_location_freshness import (
        last_seen_seconds_from_location_fields,
    )
    from services.geolocation.device_health import read_device_health
    from services.geolocation.presence import (
        apply_device_health_override,
        compute_location_status,
        presence_status_from_location_status,
    )
    from services.realtime.live_driver_status import (
        resolve_driver_status_for_fanout,
        resolve_mission_status_for_driver,
        sanitize_fanout_mission_id,
    )
    from services.realtime.socketio import fanout_driver_location_update

    driver_id_obj = envelope.get("driver_id")
    if not isinstance(driver_id_obj, (int, str)):
        return
    driver_id = int(driver_id_obj)

    p = envelope.get("payload")
    if not isinstance(p, dict):
        return

    company_id_obj = envelope.get("company_id")
    company_id: int | None = (
        int(company_id_obj) if isinstance(company_id_obj, int) else None
    )

    app = get_flask_app()
    with app.app_context():
        driver = db.session.get(Driver, driver_id)
        if company_id is None and driver is not None:
            driver_company_id = getattr(driver, "company_id", None)
            if isinstance(driver_company_id, (int, str)):
                company_id = int(driver_company_id)
        if company_id is None or company_id <= 0:
            logger.debug(
                "[processed_fanout] skip driver_id=%s reason=no_company_id",
                driver_id,
            )
            return

        try:
            lat_val = p.get("latitude", p.get("lat"))
            lon_val = p.get("longitude", p.get("lon"))
            persist_result = envelope.get("persist_result")
            if isinstance(persist_result, dict):
                snapped_lat = persist_result.get("snapped_lat")
                snapped_lon = persist_result.get("snapped_lon")
                if snapped_lat is not None and snapped_lon is not None:
                    lat_val = snapped_lat
                    lon_val = snapped_lon
            if lat_val is None or lon_val is None:
                return
            lat = float(lat_val)
            lon = float(lon_val)
        except (TypeError, ValueError):
            logger.debug(
                "[processed_fanout] skip driver_id=%s reason=bad_coords", driver_id
            )
            return

        speed = float(p.get("speed_mps", p.get("speed", 0.0)) or 0.0)
        heading = float(p.get("heading", 0.0) or 0.0)
        accuracy = float(p.get("accuracy_m", p.get("accuracy", 0.0)) or 0.0)
        recorded_at = str(
            p.get("recorded_at") or p.get("timestamp") or datetime.now(UTC).isoformat()
        )
        sent_at = str(p.get("sent_at") or datetime.now(UTC).isoformat())
        location_mode = str(p.get("location_mode") or "mission_live")
        is_background = bool(p.get("is_background", False))
        mission_id_raw = p.get("mission_id")
        mission_id_parsed: int | None = (
            int(mission_id_raw)
            if isinstance(mission_id_raw, int)
            else (
                int(mission_id_raw)
                if isinstance(mission_id_raw, str) and mission_id_raw.isdigit()
                else None
            )
        )
        mission_id = sanitize_fanout_mission_id(driver_id, mission_id_parsed)

        validated_ms = envelope.get("validated_at_ms")
        received_at = (
            _iso_from_ms(validated_ms)
            or _iso_from_ms(envelope.get("received_at_ms"))
            or datetime.now(UTC).isoformat()
        )

        last_seen_seconds = last_seen_seconds_from_location_fields(
            {
                "recorded_at": recorded_at,
                "received_at": received_at,
                "ts": p.get("ts"),
            }
        )
        location_status = compute_location_status(
            mode=location_mode, last_seen_seconds=last_seen_seconds
        )
        presence_status = presence_status_from_location_status(location_status)

        from ext import redis_client as _redis_client

        device_health = read_device_health(_redis_client, driver_id)
        presence_status, location_status = apply_device_health_override(
            presence_status,
            location_status,
            device_health,
        )

        mission_status_resolved = resolve_mission_status_for_driver(driver_id)
        is_active = (
            bool(getattr(driver, "is_active", True)) if driver is not None else True
        )
        driver_status_resolved = resolve_driver_status_for_fanout(
            mission_status=mission_status_resolved,
            is_active=is_active,
            presence_status=presence_status,
        )

        first_name = None
        last_name = None
        if driver is not None and hasattr(driver, "user") and driver.user is not None:
            first_name = getattr(driver.user, "first_name", None)
            last_name = getattr(driver.user, "last_name", None)

        trace_id = envelope.get("trace_id")
        event_id = None
        if isinstance(trace_id, str) and trace_id.strip():
            event_id = trace_id.strip()
        else:
            tei = p.get("tracking_event_id")
            if isinstance(tei, str) and tei.strip():
                event_id = tei.strip()

        canonical_payload: dict[str, Any] = {
            "driver_id": driver_id,
            "company_id": company_id,
            "lat": lat,
            "lon": lon,
            "lng": lon,
            "speed": speed,
            "speed_mps": speed,
            "heading": heading,
            "accuracy": accuracy,
            "accuracy_m": accuracy,
            "ts": recorded_at,
            "timestamp": recorded_at,
            "recorded_at": recorded_at,
            "sent_at": sent_at,
            "received_at": received_at,
            "is_background": is_background,
            "mission_id": mission_id,
            "location_mode": location_mode,
            "last_seen_seconds": last_seen_seconds,
            "location_status": location_status,
            "presence_status": presence_status,
            "status": driver_status_resolved,
            "mission_status": (
                mission_status_resolved if mission_status_resolved != "NONE" else None
            ),
            "is_available": driver_status_resolved == "available",
            "offline_reason": "",
            "source": f"kafka_processed:{envelope.get('source', 'unknown')}",
            "first_name": first_name,
            "last_name": last_name,
        }
        if device_health is not None:
            canonical_payload["device_health"] = device_health
        if event_id:
            canonical_payload["event_id"] = event_id

        persist_result = envelope.get("persist_result")
        accept_status = "accepted_observability_only"
        if isinstance(persist_result, dict):
            pr_status = persist_result.get("accept_status")
            if isinstance(pr_status, str) and pr_status.strip():
                accept_status = pr_status.strip()

        fanout_driver_location_update(
            company_id,
            canonical_payload,
            canonical_payload,
            accept_status=accept_status,
        )
        try:
            from services.monitoring.driver_location_metrics import (
                inc_tracking_fanout_emit,
            )

            inc_tracking_fanout_emit(emitter="backend_fanout")
        except Exception:
            logger.debug(
                "[processed_fanout] fanout emit metric unavailable", exc_info=True
            )


class ProcessedLocationFanoutConsumer:
    def __init__(self) -> None:
        super().__init__()
        self._consumer = None
        self._running = False
        self._initialized = False
        self._last_lag_publish_ts = 0.0
        signal.signal(signal.SIGTERM, self._shutdown_signal)
        signal.signal(signal.SIGINT, self._shutdown_signal)
        if (
            KAFKA_ENABLED
            and TRACKING_INGEST_ASYNC_ENABLED
            and TRACKING_PROCESSED_FANOUT_ENABLED
        ):
            self._init_consumer()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def _init_consumer(self) -> None:
        try:
            from kafka import KafkaConsumer as KC

            from services.kafka.bootstrap_retry import run_with_kafka_bootstrap_retry

            def _connect():
                consumer = KC(
                    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                    group_id=KAFKA_PROCESSED_FANOUT_GROUP,
                    enable_auto_commit=False,
                    auto_offset_reset=KAFKA_AUTO_OFFSET_RESET,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    key_deserializer=lambda k: k.decode("utf-8") if k else None,
                    consumer_timeout_ms=1000,
                    **_kafka_security_config(),
                )
                consumer.subscribe([TOPIC_DRIVER_LOCATION_PROCESSED])
                return consumer

            self._consumer = run_with_kafka_bootstrap_retry(
                operation_label="[processed_fanout]",
                logger=logger,
                fn=_connect,
            )
            self._initialized = True
            logger.info(
                "[processed_fanout] subscribed topic=%s group=%s",
                TOPIC_DRIVER_LOCATION_PROCESSED,
                KAFKA_PROCESSED_FANOUT_GROUP,
            )
        except ImportError:
            logger.error("[processed_fanout] kafka-python dependency missing")
        except Exception as exc:
            logger.exception("[processed_fanout] initialization failed")
            try:
                from shared.sentry_init import capture_kafka_error

                capture_kafka_error(exc)
            except Exception:
                logger.debug("[processed_fanout] sentry capture skipped", exc_info=True)

    def _shutdown_signal(self, signum, _frame) -> None:
        logger.info("[processed_fanout] shutdown signal=%s", signum)
        self._running = False

    def start(self) -> None:
        if not self._initialized or self._consumer is None:
            logger.error("[processed_fanout] not initialized")
            return
        self._running = True
        logger.info("[processed_fanout] loop start")
        from kafka.structs import OffsetAndMetadata

        from services.monitoring.driver_location_metrics import (
            inc_tracking_processed_fanout_failure,
        )

        while self._running:
            try:
                polled = self._consumer.poll(timeout_ms=1000)
                for tp, records in polled.items():
                    last_ok: OffsetAndMetadata | None = None
                    for record in records:
                        try:
                            msg = record.value
                            if isinstance(msg, dict):
                                _fanout_processed_message(msg)
                            else:
                                logger.debug(
                                    "[processed_fanout] skip non-dict partition=%s offset=%s",
                                    record.partition,
                                    record.offset,
                                )
                            last_ok = OffsetAndMetadata(record.offset + 1, "", -1)
                        except Exception as exc:
                            drv = (
                                record.value.get("driver_id")
                                if isinstance(record.value, dict)
                                else None
                            )
                            logger.exception(
                                "[processed_fanout] fanout failed partition=%s offset=%s driver_id=%s error=%s",
                                record.partition,
                                record.offset,
                                drv,
                                exc,
                            )
                            try:
                                inc_tracking_processed_fanout_failure(
                                    error_type=type(exc).__name__
                                )
                            except Exception:
                                logger.debug(
                                    "[processed_fanout] fanout failure metric unavailable",
                                    exc_info=True,
                                )
                            break
                    if last_ok is not None:
                        try:
                            self._consumer.commit({tp: last_ok})
                        except Exception:
                            logger.exception(
                                "[processed_fanout] commit failed after batch partition=%s",
                                getattr(tp, "partition", "?"),
                            )
                self._maybe_publish_lag()
            except Exception as exc:
                from shared.sentry_init import (
                    capture_kafka_error,
                    is_kafka_connection_error,
                )

                if is_kafka_connection_error(exc):
                    capture_kafka_error(exc)
                logger.exception("[processed_fanout] poll loop error")
                time.sleep(1.0)
        self.close()

    def _maybe_publish_lag(self) -> None:
        """Publie le lag consumer par partition (P1-1b), throttlé ~15 s.

        ``lag = end_offset - position`` (lag « prêt à traiter », évite le RPC
        ``committed()``). Best-effort : toute erreur est avalée.
        """
        if not TRACKING_KAFKA_LAG_METRIC_ENABLED or self._consumer is None:
            return
        now = time.time()
        if now - self._last_lag_publish_ts < TRACKING_KAFKA_LAG_METRIC_INTERVAL_S:
            return
        self._last_lag_publish_ts = now
        try:
            from services.monitoring.driver_location_metrics import (
                set_tracking_kafka_consumer_lag,
            )

            assignment = self._consumer.assignment()
            if not assignment:
                return
            end_offsets = self._consumer.end_offsets(list(assignment))
            for tp in assignment:
                position = self._consumer.position(tp)
                end_offset = end_offsets.get(tp)
                if position is None or end_offset is None:
                    continue
                set_tracking_kafka_consumer_lag(
                    group=KAFKA_PROCESSED_FANOUT_GROUP,
                    topic=tp.topic,
                    partition=tp.partition,
                    lag=end_offset - position,
                )
        except Exception:
            logger.debug("[processed_fanout] lag metric publish failed", exc_info=True)

    def close(self) -> None:
        if self._consumer is not None:
            self._consumer.close()


def run_processed_location_fanout_consumer() -> None:
    from services.monitoring.standalone_prometheus_server import (
        start_standalone_prometheus_server,
    )
    from shared.logging_utils import configure_kafka_log_noise
    from shared.sentry_init import init_sentry

    configure_kafka_log_noise()
    init_sentry()
    start_standalone_prometheus_server()
    if not (
        KAFKA_ENABLED
        and TRACKING_INGEST_ASYNC_ENABLED
        and TRACKING_PROCESSED_FANOUT_ENABLED
    ):
        logger.info(
            "[processed_fanout] disabled (need KAFKA_ENABLED, TRACKING_INGEST_ASYNC_ENABLED, TRACKING_PROCESSED_FANOUT_ENABLED), exiting cleanly"
        )
        sys.exit(0)
    from services.tracking.worker_bootstrap import validate_tracking_worker_env

    validate_tracking_worker_env("processed_fanout")
    consumer = ProcessedLocationFanoutConsumer()
    if not consumer.initialized:
        sys.exit(1)
    consumer.start()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    run_processed_location_fanout_consumer()
