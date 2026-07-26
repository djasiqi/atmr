"""Webhook Brevo transactionnel — Bearer + idempotence atomique (Lot 1)."""

from __future__ import annotations

import logging

from flask import Blueprint, request
from flask_restx import Namespace, Resource

from services.notifications.brevo_webhook import (
    process_brevo_webhook_event,
    require_brevo_webhook_secret_in_production,
    verify_brevo_bearer,
)

logger = logging.getLogger(__name__)

webhooks_brevo_bp = Blueprint("webhooks_brevo", __name__)
webhooks_brevo_ns = Namespace(
    "webhooks",
    description="Webhooks fournisseurs (Brevo)",
)


def handle_brevo_webhook() -> tuple[dict, int]:
    try:
        require_brevo_webhook_secret_in_production()
    except RuntimeError as e:
        logger.error("[brevo_webhook] %s", e)
        return {"error": "webhook_misconfigured"}, 503

    if not verify_brevo_bearer(request.headers.get("Authorization")):
        return {"error": "unauthorized"}, 401

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return {"error": "invalid_payload"}, 400

    try:
        result = process_brevo_webhook_event(payload)
    except Exception:
        logger.exception("[brevo_webhook] erreur traitement")
        return {"error": "processing_failed"}, 500

    return {
        "status": result.get("status"),
        "reason": result.get("reason"),
    }, int(result.get("http_status", 200))


@webhooks_brevo_bp.route("/webhooks/brevo", methods=["POST"])
def brevo_webhook_blueprint():
    body, status = handle_brevo_webhook()
    return body, status


@webhooks_brevo_ns.route("/brevo")
class BrevoWebhook(Resource):
    def post(self):
        return handle_brevo_webhook()
