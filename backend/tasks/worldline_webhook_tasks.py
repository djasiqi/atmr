"""Traitement asynchrone des webhooks Worldline (optionnel, voir WORLDLINE_WEBHOOK_ASYNC)."""

from __future__ import annotations

import base64
import logging
from typing import Any

from celery_app import celery, get_flask_app

logger = logging.getLogger(__name__)


@celery.task(
    bind=True,
    name="worldline.process_webhook",
    max_retries=5,
    default_retry_delay=15,
)
def process_worldline_webhook_task(
    self: Any,
    body_b64: str,
    headers_list: list[list[str]],
) -> None:
    """Décode le corps, rejoue la même logique que le handler HTTP synchrone."""
    from worldline.connect.sdk.webhooks.signature_validation_exception import (
        SignatureValidationException,
    )

    from services.worldline.webhook_service import process_webhook_request

    app = get_flask_app()
    with app.app_context():
        try:
            raw = base64.b64decode(body_b64.encode("ascii"))
            pairs = [(str(pair[0]), str(pair[1])) for pair in headers_list]
            process_webhook_request(body=raw, wsgi_headers=pairs)
        except SignatureValidationException:
            logger.warning("Worldline webhook async: signature invalide")
            return
        except Exception as exc:
            logger.exception("Worldline webhook async: échec traitement")
            try:
                raise self.retry(exc=exc) from exc
            except self.MaxRetriesExceededError:
                logger.error(
                    "Worldline webhook async: abandon après %s tentatives",
                    self.max_retries,
                )
                raise
