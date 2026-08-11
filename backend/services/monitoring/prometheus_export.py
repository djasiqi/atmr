"""Helpers export Prometheus — mode multiprocess Gunicorn safe."""

from __future__ import annotations

import os
from typing import Any


def generate_prometheus_latest() -> tuple[Any, str]:
    """Retourne (payload bytes|str, content_type).

    Si ``PROMETHEUS_MULTIPROC_DIR`` est défini : ``CollectorRegistry`` +
    ``MultiProcessCollector`` (agrégation multi-workers).
    Sinon : registre process-local (dev / worker unique).
    """
    from prometheus_client import (  # pyright: ignore[reportMissingImports]
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        generate_latest,
        multiprocess,
    )

    multiproc_dir = (os.environ.get("PROMETHEUS_MULTIPROC_DIR") or "").strip()
    if multiproc_dir:
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return generate_latest(registry), CONTENT_TYPE_LATEST
    return generate_latest(), CONTENT_TYPE_LATEST
