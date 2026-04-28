"""Automatisation facturation transport (V3) — hooks Celery / batch (progressif)."""

from __future__ import annotations

import logging
from typing import Any

from celery_app import celery

logger = logging.getLogger(__name__)


@celery.task(name="invoices.monthly_transport_billing_reminder")
def monthly_transport_billing_reminder_task() -> dict[str, Any]:
    """Tic mensuel (1er du mois) : emplacement pour rappels / futur auto-billing.

    Ne crée ni ne modifie aucune facture (sécurité). Journalisation seule.
    """
    logger.info(
        "monthly_transport_billing_reminder_task: noop (V3) — "
        "brancher ici notifications / pré-liste « factures à générer »"
    )
    return {"ok": True, "mode": "noop"}


@celery.task(
    name="invoices.batch_generate_drafts_for_period",
    task_time_limit=3600,
    task_soft_time_limit=3500,
)
def batch_generate_drafts_for_period_task(
    company_id: int, period_year: int, period_month: int
) -> dict[str, Any]:
    """Emplacement V3 « générer tout » : exécuter uniquement si activé côté API / feature flag.

    Par défaut : aucun brouillon (éviter double facturation sans garde-fous métier).
    """
    logger.info(
        "batch_generate_drafts_for_period_task company_id=%s y=%s m=%s — stub V3 (0 créé)",
        company_id,
        period_year,
        period_month,
    )
    return {
        "ok": True,
        "created_invoice_ids": [],
        "errors": [],
        "message": "Stub V3 : brancher GenerateInvoice en boucle avec idempotence / verrous",
    }
