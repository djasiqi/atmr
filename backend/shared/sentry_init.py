"""Initialisation Sentry partagée (Flask, Celery, consumers Kafka)."""

from __future__ import annotations

import os
from typing import Any

import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.flask import FlaskIntegration
from sentry_sdk.integrations.logging import ignore_logger

_DROP_EXCEPTION_TYPES = frozenset(
    {"ExpiredSignatureError", "NoAuthorizationError", "InvalidHeaderError"}
)

# Déconnexion WebSocket gevent (normal) — ne pas alerter Sentry sur /socket.io/
_SOCKET_IO_BENIGN_EXC = frozenset({"StopIteration", "GreenletExit"})

# Concurrence gevent sur sockets partagés (client déconnecté, pas de requête HTTP lue)
_GEVENT_INFRA_EXC = frozenset({"ConcurrentObjectUseError"})

_KAFKA_ERROR_TYPES = frozenset(
    {"NoBrokersAvailable", "KafkaTimeoutError", "KafkaConnectionError"}
)

# kafka-python journalise chaque retry bootstrap en ERROR (NodeNotReady, DNS, etc.).
_KAFKA_PYTHON_LOGGERS = (
    "kafka.net.manager",
    "kafka.net.inet",
    "kafka.net.selector",
    "kafka.cluster",
    "kafka.conn",
    "kafka.client",
    "kafka.consumer",
    "kafka.producer",
    "kafka.coordinator",
    "kafka.coordinator.heartbeat",
)

_KAFKA_BOOTSTRAP_LOG_MARKERS = (
    "bootstrap attempt to bootstrap-",
    "nodenotreadyerror",
    "dns resolution failure",
    "dns lookup failed for kafka-broker",
    "metadata refresh: failed",
    "temporary failure in name resolution",
    "rebalanceinprogresserror",
    "heartbeat failed for group",
    "error sending heartbeatrequest",
    "task is already done",
    "invalid file descriptor",
)


def _is_socket_io_benign_disconnect(
    event: dict[str, Any], exc_type: type[BaseException] | None
) -> bool:
    if exc_type is None or exc_type.__name__ not in _SOCKET_IO_BENIGN_EXC:
        return False
    req_url = (event.get("request") or {}).get("url", "")
    return "/socket.io/" in req_url


def _is_kafka_bootstrap_log_noise(message: str, logger_name: str) -> bool:
    if logger_name.startswith("kafka."):
        return True
    lowered = message.lower()
    return any(marker in lowered for marker in _KAFKA_BOOTSTRAP_LOG_MARKERS)


def _is_gevent_infrastructure_noise(
    event: dict[str, Any],
    exc_type: type[BaseException] | None,
    message: str,
) -> bool:
    if exc_type is not None:
        if exc_type.__name__ in _GEVENT_INFRA_EXC:
            return True
        if _is_socket_io_benign_disconnect(event, exc_type):
            return True
    if "already used by another greenlet" in message:
        return True
    if "Error handling request (no URI read)" in message:
        return True
    return False


def before_send(event: dict[str, Any], hint: dict[str, Any] | None) -> dict[str, Any] | None:
    """Filtre le bruit Sentry (JWT expirés, déconnexions / concurrence gevent)."""
    exc_info = hint.get("exc_info") if hint else None
    logentry = event.get("logentry") or {}
    message = str(logentry.get("message") or event.get("message") or "")
    logger_name = str(event.get("logger") or "")

    if _is_kafka_bootstrap_log_noise(message, logger_name):
        return None

    if exc_info:
        exc_type = exc_info[0]
        if exc_type is not None:
            exc_name = exc_type.__name__
            if exc_name in _DROP_EXCEPTION_TYPES:
                return None
            # Race kafka-python au shutdown / rebalance (non bloquant)
            if exc_name == "ValueError" and "invalid file descriptor" in message.lower():
                return None
            if exc_name == "RuntimeError" and "task is already done" in message.lower():
                return None
        if _is_gevent_infrastructure_noise(event, exc_type, message):
            return None

    # logger.exception() sur routes notifications : JWT expiré remonté comme erreur 500
    if "Signature has expired" in message and (
        "InstitutionNotifications" in message or "CompanyNotifications" in message
    ):
        return None
    if "Accès conversation refusé" in message and (
        "mark_read" in message or "messages_hub" in message
    ):
        return None

    if _is_gevent_infrastructure_noise(event, None, message):
        return None

    return event


def init_sentry(
    *,
    environment: str | None = None,
    flask: bool = False,
    celery: bool = False,
) -> None:
    """Initialise Sentry une seule fois par processus."""
    dsn = os.getenv("SENTRY_DSN")
    env = environment or os.getenv("FLASK_CONFIG") or os.getenv("FLASK_ENV") or "development"
    if not dsn or env == "testing":
        return

    if sentry_sdk.Hub.current.client is not None:
        return

    for kafka_logger in _KAFKA_PYTHON_LOGGERS:
        ignore_logger(kafka_logger)

    integrations: list[Any] = []
    if flask:
        integrations.append(FlaskIntegration())
    if celery:
        integrations.append(CeleryIntegration())

    _sentry_traces_default = "0.1" if env == "production" else "1.0"
    sentry_sdk.init(
        dsn=dsn,
        integrations=integrations,
        traces_sample_rate=float(
            os.getenv("SENTRY_TRACES_SAMPLE_RATE", _sentry_traces_default)
        ),
        environment=env,
        release=os.getenv("SENTRY_RELEASE") or os.getenv("GIT_SHA"),
        before_send=before_send,
    )


def capture_kafka_error(exc: BaseException) -> None:
    """Remonte explicitement les erreurs Kafka à Sentry."""
    if exc.__class__.__name__ in _KAFKA_ERROR_TYPES:
        sentry_sdk.capture_exception(exc)
        return
    message = str(exc).lower()
    if any(token in message for token in ("nobrokersavailable", "kafkatimeout", "kafkaconnection")):
        sentry_sdk.capture_exception(exc)


def is_kafka_connection_error(exc: BaseException) -> bool:
    """True si l'exception ressemble à une erreur de connexion Kafka."""
    if exc.__class__.__name__ in _KAFKA_ERROR_TYPES:
        return True
    message = str(exc).lower()
    return any(token in message for token in ("nobrokersavailable", "kafkatimeout", "kafkaconnection"))
