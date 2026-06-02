"""Initialisation Sentry partagée (Flask, Celery, consumers Kafka)."""

from __future__ import annotations

import os
from typing import Any

import sentry_sdk
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.flask import FlaskIntegration

_DROP_EXCEPTION_TYPES = frozenset(
    {"ExpiredSignatureError", "NoAuthorizationError", "InvalidHeaderError"}
)

_KAFKA_ERROR_TYPES = frozenset(
    {"NoBrokersAvailable", "KafkaTimeoutError", "KafkaConnectionError"}
)


def before_send(event: dict[str, Any], hint: dict[str, Any] | None) -> dict[str, Any] | None:
    """Filtre le bruit Sentry (JWT expirés, StopIteration Socket.IO)."""
    exc_info = hint.get("exc_info") if hint else None
    if not exc_info:
        return event

    exc_type = exc_info[0]
    if exc_type is None:
        return event

    if exc_type.__name__ in _DROP_EXCEPTION_TYPES:
        return None

    if exc_type is StopIteration:
        req_url = (event.get("request") or {}).get("url", "")
        if "/socket.io/" in req_url:
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
