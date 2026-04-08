"""Surveillance périodique des paiements Worldline bloqués en PENDING."""

from __future__ import annotations

import logging
import os

from celery_app import celery, get_flask_app

logger = logging.getLogger(__name__)


@celery.task(name="worldline.reconcile_stale_pending")
def reconcile_stale_worldline_pending_task() -> dict:
    """Log un avertissement si des PENDING Worldline stagnent (webhook manquant, abandon, etc.)."""
    if os.getenv("WORLDLINE_RECONCILIATION_BEAT", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return {"skipped": True, "reason": "WORLDLINE_RECONCILIATION_BEAT désactivé"}

    app = get_flask_app()
    with app.app_context():
        from services.worldline.reconciliation import summarize_stale_worldline_pending

        summary = summarize_stale_worldline_pending()
        if summary["count"] > 0:
            logger.warning(
                "Worldline: %s paiement(s) PENDING stagnants (hosted checkout créé, pas de webhook final)",
                summary["count"],
                extra={
                    "worldline_stale_pending_count": summary["count"],
                    "worldline_stale_payment_ids": summary["payment_ids"][:50],
                },
            )
        return summary
