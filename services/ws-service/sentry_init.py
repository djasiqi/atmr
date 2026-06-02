"""Sentry minimal pour ws-service (processus standalone)."""

from __future__ import annotations

import os

import sentry_sdk


def init_ws_sentry() -> None:
    dsn = os.getenv("SENTRY_DSN")
    env = os.getenv("SENTRY_ENVIRONMENT") or os.getenv("FLASK_ENV") or "production"
    if not dsn or env == "testing":
        return
    if sentry_sdk.Hub.current.client is not None:
        return
    _traces_default = "0.1" if env == "production" else "1.0"
    sentry_sdk.init(
        dsn=dsn,
        environment=env,
        traces_sample_rate=float(
            os.getenv("SENTRY_TRACES_SAMPLE_RATE", _traces_default)
        ),
        release=os.getenv("SENTRY_RELEASE") or os.getenv("GIT_SHA"),
    )


def capture_kafka_error(exc: BaseException) -> None:
    name = exc.__class__.__name__
    if name in {"NoBrokersAvailable", "KafkaTimeoutError", "KafkaConnectionError"}:
        sentry_sdk.capture_exception(exc)
        return
    message = str(exc).lower()
    if any(token in message for token in ("nobrokersavailable", "kafkatimeout", "kafkaconnection")):
        sentry_sdk.capture_exception(exc)
