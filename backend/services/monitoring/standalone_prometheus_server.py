"""Serveur HTTP Prometheus minimal pour workers sans Flask (consumers Kafka)."""

from __future__ import annotations

import logging
import os
import threading

logger = logging.getLogger(__name__)

_started = False
_process_collector_registered = False
_lock = threading.Lock()


def start_standalone_prometheus_server() -> None:
    """Démarre ``prometheus_client.start_http_server`` (idempotent, non bloquant)."""
    global _started, _process_collector_registered
    with _lock:
        if _started:
            return
        enabled = os.getenv("STANDALONE_PROMETHEUS_ENABLED", "true").lower() not in (
            "0",
            "false",
            "no",
            "off",
        )
        if not enabled:
            logger.info(
                "[prometheus_standalone] désactivé (STANDALONE_PROMETHEUS_ENABLED=false)"
            )
            return
        port = int(os.getenv("STANDALONE_PROMETHEUS_PORT", "9115"))
        try:
            from prometheus_client import ProcessCollector, start_http_server

            # P2-3 : process_resident_memory_bytes (ProcessCollector s'auto-enregistre ;
            # ignorer si déjà présent dans le registry par défaut).
            if not _process_collector_registered:
                try:
                    ProcessCollector()
                    _process_collector_registered = True
                except ValueError:
                    _process_collector_registered = True

            start_http_server(port, addr="0.0.0.0")
            _started = True
            logger.info("[prometheus_standalone] écoute 0.0.0.0:%s/metrics", port)
        except ImportError:
            logger.warning(
                "[prometheus_standalone] prometheus_client absent — métriques HTTP désactivées"
            )
        except Exception:
            logger.exception("[prometheus_standalone] démarrage impossible")
