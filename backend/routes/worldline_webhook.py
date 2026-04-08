"""Webhook Worldline Connect (hors JWT — vérification HMAC SDK)."""

from __future__ import annotations

import base64
import logging
import os

from flask import Blueprint, Response, request

from services.worldline.webhook_service import (
    process_webhook_request,
    webhook_helper_configured,
)

logger = logging.getLogger(__name__)

worldline_webhook_bp = Blueprint("worldline_webhook", __name__)


@worldline_webhook_bp.route("/payments/worldline/webhook", methods=["GET", "POST"])
def worldline_webhook() -> Response | tuple[str, int]:
    """GET: validation d'endpoint (Configuration Center). POST: événements."""
    if request.method == "GET":
        challenge = request.headers.get("X-GCS-Webhooks-Endpoint-Verification", "")
        return challenge, 200

    if not webhook_helper_configured():
        logger.error("Worldline webhook appelé mais clés webhooks non configurées")
        return Response("webhook not configured", status=503)

    body = request.get_data(cache=False, as_text=False) or b""

    async_flag = os.getenv("WORLDLINE_WEBHOOK_ASYNC", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )
    if async_flag:
        from tasks.worldline_webhook_tasks import process_worldline_webhook_task

        hdrs = [
            [str(k), str(v)] for k, v in request.headers.items() if v is not None
        ]
        process_worldline_webhook_task.delay(
            base64.b64encode(body).decode("ascii"),
            hdrs,
        )
        return Response(status=204)

    try:
        process_webhook_request(body=body, wsgi_headers=request.headers.items())
    except Exception as e:
        from worldline.connect.sdk.webhooks.signature_validation_exception import (
            SignatureValidationException,
        )

        if isinstance(e, SignatureValidationException):
            logger.warning("Worldline webhook signature invalide")
            return Response("invalid signature", status=400)
        logger.exception("Worldline webhook processing failed")
        return Response("internal error", status=500)

    return Response(status=204)
