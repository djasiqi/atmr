"""Régénération FORCÉE du PDF d'une facture partenaire.

Même contrat que le catalogue standard, catalogue isolé :
relecture DB partenaire → lignes snapshot / destinataire live → nouveau fichier
→ remplacement de pdf_url seulement après succès. Échec = ancien PDF conservé.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from application.invoices.force_regenerate_invoice_pdf import (
    _delete_replaced_invoice_pdf,
)
from application.invoices.partner_invoice_access import load_accessible_partner_invoice
from application.invoices.partner_invoice_lines import (
    ensure_partner_invoice_lines,
    line_amounts_for_pdf,
    load_partner_invoice_transfers,
)
from ext import db
from models.partner_invoice import PartnerInvoiceStatus
from services.partnerships.invoices import PartnerInvoiceService

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ForceRegeneratePartnerInvoicePdfResult:
    ok: bool
    pdf_url: str | None = None
    previous_pdf_url: str | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


_BLOCKED_STATUSES = {PartnerInvoiceStatus.CANCELLED, PartnerInvoiceStatus.PAID}


def force_regenerate_partner_invoice_pdf(
    *, company_id: int, partner_invoice_id: int
) -> ForceRegeneratePartnerInvoicePdfResult:
    """Point d'entrée unique partenaire. Ne jamais appeler force_regenerate_invoice_pdf."""
    db.session.expire_all()
    partner_invoice = load_accessible_partner_invoice(company_id, partner_invoice_id)
    if partner_invoice is None:
        return ForceRegeneratePartnerInvoicePdfResult(
            ok=False,
            error={"error": "Facture partenaire introuvable"},
            status_code=404,
        )
    if partner_invoice.status in _BLOCKED_STATUSES:
        return ForceRegeneratePartnerInvoicePdfResult(
            ok=False,
            previous_pdf_url=partner_invoice.pdf_url,
            error={
                "error": (
                    "Impossible de régénérer le PDF: la facture partenaire est "
                    f"{partner_invoice.status}."
                )
            },
            status_code=400,
        )

    previous_pdf_url = partner_invoice.pdf_url
    ensure_partner_invoice_lines(partner_invoice)
    transfers = load_partner_invoice_transfers(partner_invoice)
    if not transfers and not partner_invoice.lines:
        return ForceRegeneratePartnerInvoicePdfResult(
            ok=False,
            previous_pdf_url=previous_pdf_url,
            error={"error": "Aucun transfert associé à cette facture partenaire."},
            status_code=400,
        )

    try:
        service = PartnerInvoiceService()
        pdf_url = service._generate_invoice_pdf(
            partner_invoice,
            transfers,
            line_amounts=line_amounts_for_pdf(partner_invoice),
        )
    except Exception:
        logger.exception(
            "Régénération PDF partenaire interrompue partner_invoice_id=%s",
            partner_invoice_id,
        )
        db.session.rollback()
        return ForceRegeneratePartnerInvoicePdfResult(
            ok=False,
            previous_pdf_url=previous_pdf_url,
            error={"error": "Erreur lors de la génération du PDF partenaire"},
            status_code=500,
        )

    if not pdf_url:
        db.session.rollback()
        return ForceRegeneratePartnerInvoicePdfResult(
            ok=False,
            previous_pdf_url=previous_pdf_url,
            error={"error": "Impossible de régénérer le PDF partenaire"},
            status_code=500,
        )

    partner_invoice.pdf_url = pdf_url
    db.session.commit()
    _delete_replaced_invoice_pdf(previous_pdf_url, pdf_url)
    return ForceRegeneratePartnerInvoicePdfResult(
        ok=True,
        pdf_url=pdf_url,
        previous_pdf_url=previous_pdf_url,
    )
