"""Tâches Celery — génération / régénération PDF factures transport (V2 async)."""

from __future__ import annotations

import logging
from typing import Any

from celery_app import celery

logger = logging.getLogger(__name__)


@celery.task(
    name="invoices.regenerate_standard_invoice_pdf",
    acks_late=True,
    task_time_limit=300,
    task_soft_time_limit=240,
    max_retries=1,
    autoretry_for=(Exception,),
)
def regenerate_standard_invoice_pdf_task(
    company_id: int, invoice_id: int
) -> dict[str, Any]:
    """Régénère le PDF d'une facture transport (hors partenaire) en file d'attente."""
    from application.invoices.force_regenerate_invoice_pdf import (
        force_regenerate_invoice_pdf,
    )

    result = force_regenerate_invoice_pdf(
        company_id=company_id, invoice_id=invoice_id
    )
    if result.ok and result.pdf_url:
        return {
            "ok": True,
            "pdf_url": result.pdf_url,
            "pdf_generated_at": result.generated_at,
        }
    return {
        "ok": False,
        "error": (result.error or {}).get("error", "Impossible de générer le PDF"),
        "status_code": result.status_code or 500,
    }
