"""Surveillance périodique des paiements Saferpay bloqués en PENDING."""

from __future__ import annotations

import logging
import os
from typing import Any

from celery_app import celery, get_flask_app

logger = logging.getLogger(__name__)


@celery.task(name="saferpay.reconcile_stale_pending")
def reconcile_stale_saferpay_pending_task() -> dict[str, Any]:
    if os.getenv("SAFERPAY_RECONCILIATION_BEAT", "1").strip().lower() in (
        "0",
        "false",
        "no",
        "off",
    ):
        return {"skipped": True, "reason": "SAFERPAY_RECONCILIATION_BEAT désactivé"}

    app = get_flask_app()
    with app.app_context():
        from services.saferpay.reconciliation import (
            run_finalize_retries_for_stale_pending,
            summarize_stale_saferpay_pending,
        )

        summary = summarize_stale_saferpay_pending()
        retry_out: dict[str, Any] | None = None
        if os.getenv("SAFERPAY_RECONCILE_RETRY", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            retry_out = run_finalize_retries_for_stale_pending()

        if summary["count"] > 0:
            logger.warning(
                "Saferpay: %s paiement(s) PENDING stagnants (session non finalisée)",
                summary["count"],
                extra={
                    "saferpay_stale_pending_count": summary["count"],
                    "saferpay_stale_payment_ids": summary["payment_ids"][:50],
                    "saferpay_retry": retry_out or {},
                },
            )
        out: dict[str, Any] = dict(summary)
        if retry_out is not None:
            out["finalize_retry"] = retry_out
        return out
