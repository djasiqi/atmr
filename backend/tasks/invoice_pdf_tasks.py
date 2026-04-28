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
    from ext import db
    from models.enums import InvoiceStatus
    from repositories.invoice_repository import InvoiceRepository

    from application.invoices.generate_invoice_pdf import GenerateInvoicePdfUseCase

    repo = InvoiceRepository()
    invoice = repo.find_model_by_id_and_company(invoice_id, company_id)
    if not invoice:
        return {
            "ok": False,
            "error": "Facture introuvable",
            "status_code": 404,
        }

    locked = {
        InvoiceStatus.SENT,
        InvoiceStatus.PARTIALLY_PAID,
        InvoiceStatus.PAID,
    }
    if invoice.status in locked:
        return {
            "ok": False,
            "error": "La facture est verrouillée; régénération PDF interdite.",
            "status_code": 400,
        }

    uc = GenerateInvoicePdfUseCase()
    pdf_result = uc.execute(invoice=invoice, force_regenerate=True)
    if pdf_result.ok and pdf_result.pdf_url:
        invoice.pdf_url = pdf_result.pdf_url
        db.session.commit()
        return {"ok": True, "pdf_url": pdf_result.pdf_url}

    if pdf_result.error:
        return {
            "ok": False,
            "error": pdf_result.error.get("error", "Erreur génération PDF"),
            "status_code": pdf_result.status_code or 500,
        }
    return {
        "ok": False,
        "error": "Impossible de générer le PDF",
        "status_code": 500,
    }
