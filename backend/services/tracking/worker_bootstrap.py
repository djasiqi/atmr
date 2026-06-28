"""Validation d'environnement pour les workers tracking Kafka (consumer, fanout).

Échoue au démarrage si les variables critiques sont absentes — évite la boucle
DLQ silencieuse (ex. APP_ENCRYPTION_KEY_B64 manquante en worker Docker).
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger(__name__)


def validate_tracking_worker_env(service_label: str) -> None:
    """Vérifie les prérequis communs avant `get_flask_app()` dans un worker."""
    missing: list[str] = []
    if not (os.getenv("APP_ENCRYPTION_KEY_B64") or "").strip():
        missing.append("APP_ENCRYPTION_KEY_B64")
    if not (os.getenv("REDIS_URL") or "").strip():
        missing.append("REDIS_URL")

    flask_config = (os.getenv("FLASK_CONFIG") or "production").strip()
    if flask_config == "production":
        cors = (os.getenv("SOCKETIO_CORS_ORIGINS") or "").strip()
        if not cors or cors == "*":
            missing.append("SOCKETIO_CORS_ORIGINS")

    if missing:
        logger.critical(
            "[%s] environnement worker incomplet: %s — "
            "aligner avec atmr_api / celery-worker (voir docker-compose.kafka.dev.yml)",
            service_label,
            ", ".join(missing),
        )
        sys.exit(1)

    try:
        from models.base import _load_encryption_key

        _load_encryption_key()
    except Exception as exc:
        logger.critical("[%s] bootstrap env invalide: %s", service_label, exc)
        sys.exit(1)

    logger.info(
        "[%s] bootstrap env OK (FLASK_CONFIG=%s REDIS_URL défini)",
        service_label,
        flask_config,
    )
