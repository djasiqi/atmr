"""Backoff exponentiel pour le bootstrap Kafka (brokers lents au démarrage)."""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, TypeVar

T = TypeVar("T")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def run_with_kafka_bootstrap_retry(
    *,
    operation_label: str,
    logger: logging.Logger,
    fn: Callable[[], T],
) -> T:
    """Exécute ``fn`` en réessayant sur erreur jusqu'à épuisement des tentatives.

    Ne corrige pas un cluster injoignable (mauvais réseau ou brokers absents) :
    après ``KAFKA_BOOTSTRAP_MAX_ATTEMPTS``, la dernière exception est propagée.

    Variables d'environnement :

    - ``KAFKA_BOOTSTRAP_MAX_ATTEMPTS`` — nombre max de tentatives (défaut ``60``, minimum ``1``).
    - ``KAFKA_BOOTSTRAP_INITIAL_BACKOFF_SEC`` — premier délai entre deux échecs (défaut ``1.0``).
    - ``KAFKA_BOOTSTRAP_MAX_BACKOFF_SEC`` — plafond du backoff exponentiel (défaut ``30.0``).
    """
    max_attempts = max(1, _env_int("KAFKA_BOOTSTRAP_MAX_ATTEMPTS", 60))
    base_sleep = max(0.05, _env_float("KAFKA_BOOTSTRAP_INITIAL_BACKOFF_SEC", 1.0))
    max_sleep = max(base_sleep, _env_float("KAFKA_BOOTSTRAP_MAX_BACKOFF_SEC", 30.0))

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_attempts:
                logger.exception(
                    "%s kafka bootstrap failed after %s attempts",
                    operation_label,
                    max_attempts,
                )
                raise
            sleep_s = min(max_sleep, base_sleep * (2 ** (attempt - 1)))
            logger.warning(
                "%s kafka bootstrap attempt %s/%s failed: %s; retry in %.2fs",
                operation_label,
                attempt,
                max_attempts,
                exc,
                sleep_s,
            )
            time.sleep(sleep_s)

    raise RuntimeError(
        f"{operation_label} kafka bootstrap retry loop exited unexpectedly"
    )
