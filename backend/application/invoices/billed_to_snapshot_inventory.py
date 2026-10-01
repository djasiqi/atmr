"""Inventaire READ-ONLY des factures figées sans snapshot (aucun UPDATE).

À exécuter en local via Docker, jamais en production depuis cette tâche.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from models.enums import InvoiceStatus

_FROZEN = (
    InvoiceStatus.SENT.value,
    InvoiceStatus.PARTIALLY_PAID.value,
    InvoiceStatus.PAID.value,
    InvoiceStatus.OVERDUE.value,
    InvoiceStatus.CANCELLED.value,
)


def count_frozen_invoices_missing_snapshots(session: Any) -> dict[str, Any]:
    """Compte, par statut, les factures hors DRAFT sans billed_to / qr_debtor."""
    rows = session.execute(
        text(
            """
            SELECT status,
                   COUNT(*) AS total,
                   COUNT(*) FILTER (
                       WHERE meta IS NULL
                          OR NOT (meta ? 'billed_to_snapshot')
                   ) AS missing_billed_to,
                   COUNT(*) FILTER (
                       WHERE meta IS NULL
                          OR NOT (meta ? 'qr_debtor_snapshot')
                   ) AS missing_qr_debtor,
                   COUNT(*) FILTER (
                       WHERE meta IS NULL
                          OR NOT (meta ? 'qr_creditor_snapshot')
                   ) AS missing_qr_creditor
            FROM invoices
            WHERE status = ANY(:statuses)
            GROUP BY status
            ORDER BY status
            """
        ),
        {"statuses": list(_FROZEN)},
    ).mappings()
    by_status = {row["status"]: dict(row) for row in rows}
    return {
        "by_status": by_status,
        "sent_without_billed_to": int(
            (by_status.get("sent") or {}).get("missing_billed_to") or 0
        ),
        "paid_without_billed_to": int(
            (by_status.get("paid") or {}).get("missing_billed_to") or 0
        ),
        "cancelled_without_billed_to": int(
            (by_status.get("cancelled") or {}).get("missing_billed_to") or 0
        ),
        "other_frozen_without_billed_to": int(
            ((by_status.get("partially_paid") or {}).get("missing_billed_to") or 0)
            + ((by_status.get("overdue") or {}).get("missing_billed_to") or 0)
        ),
    }
