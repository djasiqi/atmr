"""Durée des routes / handlers — logs structurés + Prometheus (plan perf v3.2.1)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar

from services.monitoring.chat_metrics import observe_route_duration_ms

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


@contextmanager
def route_duration_span(route: str, **extra: Any):
    started = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = int((time.perf_counter() - started) * 1000)
        observe_route_duration_ms(route, duration_ms)
        logger.info(
            "route_duration",
            extra={
                "route": route,
                "duration_ms": duration_ms,
                **extra,
            },
        )


def timed_route(route: str) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any):
            with route_duration_span(route):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
