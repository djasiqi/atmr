"""Paiements Saferpay PENDING avec session non finalisée (observabilité)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from ext import db
from models.enums import PaymentStatus
from models.payment import Payment

logger = logging.getLogger(__name__)


def summarize_stale_saferpay_pending(
    *,
    min_age_minutes: int = 30,
    limit: int = 500,
) -> dict[str, Any]:
    if min_age_minutes < 1:
        raise ValueError("min_age_minutes doit être >= 1")
    cutoff = datetime.now(UTC) - timedelta(minutes=min_age_minutes)
    rows = (
        db.session.query(Payment)
        .filter(
            Payment.payment_provider == "saferpay",
            Payment.status == PaymentStatus.PENDING,
            Payment.saferpay_token.isnot(None),
            Payment.updated_at < cutoff,
        )
        .order_by(Payment.updated_at.asc())
        .limit(limit)
        .all()
    )
    items = [
        {
            "payment_id": r.id,
            "booking_id": r.booking_id,
            "updated_at": (
                r.updated_at.isoformat()
                if getattr(r, "updated_at", None) is not None
                else None
            ),
        }
        for r in rows
    ]
    return {
        "count": len(items),
        "payment_ids": [i["payment_id"] for i in items],
        "items": items,
    }


def run_finalize_retries_for_stale_pending(
    *,
    min_age_minutes: int = 30,
    max_attempts: int = 15,
) -> dict[str, Any]:
    """Appelle ``finalize_saferpay_payment`` sur les paiements PENDING stagnants (quota strict).

    Activé côté Celery si ``SAFERPAY_RECONCILE_RETRY`` vaut 1/true/on.
    """
    from services.saferpay.assert_response_status import SAFERPAY_FINALIZE_COMPLETED
    from services.saferpay.finalize_payment import finalize_saferpay_payment

    summary = summarize_stale_saferpay_pending(
        min_age_minutes=min_age_minutes,
        limit=max(max_attempts * 2, 50),
    )
    attempted = 0
    completed = 0
    errors = 0
    for item in summary["items"][:max_attempts]:
        pid = item["payment_id"]
        row = db.session.get(Payment, pid)
        if row is None:
            continue
        if getattr(row, "payment_provider", None) != "saferpay":
            continue
        try:
            out = finalize_saferpay_payment(row)
            attempted += 1
            if out.get("status") == SAFERPAY_FINALIZE_COMPLETED:
                completed += 1
        except Exception:
            errors += 1
            db.session.rollback()
            logger.exception("Saferpay reconcile finalize payment_id=%s", pid)
    return {
        "stale_summary_count": summary["count"],
        "retry_attempted": attempted,
        "retry_completed": completed,
        "retry_errors": errors,
    }
