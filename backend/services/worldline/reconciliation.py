"""Détection des paiements Worldline restés en PENDING (observabilité / suivi)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from ext import db
from models.enums import PaymentStatus
from models.payment import Payment


def list_stale_worldline_pending_payments(
    *,
    min_age_minutes: int = 30,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Paiements Worldline encore PENDING avec session MyCheckout, non mis à jour depuis min_age_minutes."""
    if min_age_minutes < 1:
        msg = "min_age_minutes doit être >= 1"
        raise ValueError(msg)
    cutoff = datetime.now(UTC) - timedelta(minutes=min_age_minutes)
    rows = (
        db.session.query(Payment)
        .filter(
            Payment.payment_provider == "worldline",
            Payment.status == PaymentStatus.PENDING,
            Payment.worldline_hosted_checkout_id.isnot(None),
            Payment.updated_at < cutoff,
        )
        .order_by(Payment.updated_at.asc())
        .limit(limit)
        .all()
    )
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "payment_id": r.id,
                "booking_id": r.booking_id,
                "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                "hosted_checkout_id": r.worldline_hosted_checkout_id,
            }
        )
    return out


def summarize_stale_worldline_pending(
    *,
    min_age_minutes: int = 30,
    limit: int = 500,
) -> dict[str, Any]:
    """Résumé pour logs, métriques ou endpoint admin futur."""
    items = list_stale_worldline_pending_payments(
        min_age_minutes=min_age_minutes,
        limit=limit,
    )
    return {
        "count": len(items),
        "payment_ids": [i["payment_id"] for i in items],
        "items": items,
    }
