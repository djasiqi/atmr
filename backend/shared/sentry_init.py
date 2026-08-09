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
    "kafka.net.transport",
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
    "error receiving network data closing socket",
)


def _is_socket_io_benign_disconnect(
    event: dict[str, Any], exc_type: type[BaseException] | None
) -> bool:
    if exc_type is None or exc_type.__name__ not in _SOCKET_IO_BENIGN_EXC:
        return False
    req_url = (event.get("request") or {}).get("url", "")
    return "/socket.io/" in req_url


def _is_kafka_bootstrap_log_noise(message: str, logger_name: str) -> bool:
    lowered = message.lower()
    if "task is already done" in lowered:
        return True
    if logger_name.startswith("kafka."):
        return True
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
    return "Error handling request (no URI read)" in message


def _log_event_message(event: dict[str, Any]) -> str:
    """Message logging complet (template + params) pour filtres Sentry."""
    logentry = event.get("logentry") or {}
    message = str(logentry.get("message") or event.get("message") or "")
    params = logentry.get("params")
    if isinstance(params, (list, tuple)):
        message = f"{message} {' '.join(str(p) for p in params)}"
    return message


_SENSITIVE_BODY_KEYS = frozenset(
    {
        "recovery_credential",
        "refresh_token",
        "access_token",
        "token",
        "revocation_secret",
        "password",
        "password_hash",
        "new_password",
        "old_password",
    }
)
_SENSITIVE_HEADER_KEYS = frozenset(
    {
        "authorization",
        "cookie",
        "x-refresh-token",
        "x-access-token",
    }
)
_REDACTED = "[Filtered]"


def _scrub_mapping(data: Any) -> Any:
    """Masque récursivement les clés sensibles dans dict/list."""
    if isinstance(data, dict):
        scrubbed: dict[Any, Any] = {}
        for key, value in data.items():
            key_str = str(key)
            if key_str.lower() in _SENSITIVE_BODY_KEYS or key_str.lower() in {
                "authorization",
                "cookie",
            }:
                scrubbed[key] = _REDACTED
            else:
                scrubbed[key] = _scrub_mapping(value)
        return scrubbed
    if isinstance(data, list):
        return [_scrub_mapping(item) for item in data]
    return data


def _scrub_request_headers(headers: Any) -> Any:
    if not isinstance(headers, dict):
        return headers
    out: dict[Any, Any] = {}
    for key, value in headers.items():
        if str(key).lower() in _SENSITIVE_HEADER_KEYS:
            out[key] = _REDACTED
        else:
            out[key] = value
    return out


def scrub_sensitive_event_data(event: dict[str, Any]) -> dict[str, Any]:
    """Retire credentials / tokens des payloads Sentry (request, extra, contexts)."""
    request = event.get("request")
    if isinstance(request, dict):
        if "data" in request:
            request["data"] = _scrub_mapping(request["data"])
        if "headers" in request:
            request["headers"] = _scrub_request_headers(request["headers"])
        if "cookies" in request:
            request["cookies"] = _REDACTED
        event["request"] = request

    if "extra" in event:
        event["extra"] = _scrub_mapping(event["extra"])

    contexts = event.get("contexts")
    if isinstance(contexts, dict):
        event["contexts"] = _scrub_mapping(contexts)

    return event


def before_send(
    event: dict[str, Any], hint: dict[str, Any] | None
) -> dict[str, Any] | None:
    """Filtre le bruit Sentry (JWT expirés, déconnexions / concurrence gevent)."""
    exc_info = hint.get("exc_info") if hint else None
    message = _log_event_message(event)
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
            if (
                exc_name == "ValueError"
                and "invalid file descriptor" in message.lower()
            ):
                return None
            if exc_name == "RuntimeError" and "task is already done" in message.lower():
                return None
        if _is_gevent_infrastructure_noise(event, exc_type, message):
            return None
    elif "task is already done" in message.lower() and logger_name.startswith("kafka."):
        return None

    # logger.exception() sur routes notifications / timeline : JWT expiré remonté comme erreur 500
    if "Signature has expired" in message and (
        "InstitutionNotifications" in message
        or "CompanyNotifications" in message
        or "[Timeline]" in message
        or "[InstitutionBilling]" in message
        or "[Export]" in message
    ):
        return None
    if "RemoteDisconnected" in message and "Google Maps" in message:
        return None
    if "Accès conversation refusé" in message and (
        "mark_read" in message or "messages_hub" in message
    ):
        return None

    if _is_gevent_infrastructure_noise(event, None, message):
        return None

    return scrub_sensitive_event_data(event)


def init_sentry(
    *,
    environment: str | None = None,
    flask: bool = False,
    celery: bool = False,
) -> None:
    """Initialise Sentry une seule fois par processus."""
    dsn = os.getenv("SENTRY_DSN")
    env = (
        environment
        or os.getenv("FLASK_CONFIG")
        or os.getenv("FLASK_ENV")
        or "development"
    )
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
    if any(
        token in message
        for token in ("nobrokersavailable", "kafkatimeout", "kafkaconnection")
    ):
        sentry_sdk.capture_exception(exc)


def is_kafka_connection_error(exc: BaseException) -> bool:
    """True si l'exception ressemble à une erreur de connexion Kafka."""
    if exc.__class__.__name__ in _KAFKA_ERROR_TYPES:
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in ("nobrokersavailable", "kafkatimeout", "kafkaconnection")
    )
