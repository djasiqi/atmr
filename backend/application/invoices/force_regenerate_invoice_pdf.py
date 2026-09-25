"""Régénération FORCÉE du PDF facture depuis l'état courant de la base.

Contrat figé (CLOSED) pour tous les points d'entrée (ligne de facture, modale
brouillon, tâche Celery) : relire la facture et ses relations, ignorer l'ancien
binaire / snapshot périmé, écrire un nouveau fichier, remplacer la référence
seulement après succès. En cas d'échec, l'ancien PDF reste servi.

Voir docs/facturation/regenerer-pdf-contrat.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.orm import joinedload, selectinload

from application.invoices.edit_draft_invoice import invoice_allows_line_editing
from application.invoices.generate_invoice_pdf import GenerateInvoicePdfUseCase
from application.invoices.invoice_pdf_state import (
    mark_pdf_ready,
    normalize_invoice_meta_dict,
)
from ext import db
from models import Client, Invoice

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ForceRegenerateInvoicePdfResult:
    ok: bool
    pdf_url: str | None = None
    generated_at: str | None = None
    previous_pdf_url: str | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


def reload_invoice_graph_for_pdf(invoice_id: int, company_id: int) -> Invoice | None:
    """Relecture obligatoire de la facture et des relations nécessaires au PDF."""
    db.session.expire_all()
    return (
        Invoice.query.options(
            joinedload(Invoice.company),
            joinedload(Invoice.client).joinedload(Client.user),
            selectinload(Invoice.lines),
            joinedload(Invoice.payments),
            joinedload(Invoice.billing_party),
            joinedload(Invoice.billed_to_company),
        )
        .filter_by(id=invoice_id, company_id=company_id)
        .first()
    )


def sync_patient_billing_party_from_live(invoice: Invoice) -> None:
    """Aligne un BillingParty PATIENT sur le client / patient actuel.

    Ne touche pas aux numéros, montants, lignes ni au type de payeur.
    Un tiers payeur (organisme) n'est jamais réécrit depuis le client.
    """
    from models.enums import BillingPartyType
    from services.documents.invoice_recipient import live_patient_payer_identity

    bp = getattr(invoice, "billing_party", None)
    if bp is None:
        return
    if getattr(bp, "type", None) != BillingPartyType.PATIENT:
        db.session.refresh(bp)
        return

    live_name, live_addr = live_patient_payer_identity(invoice)
    changed = False
    if live_name and (bp.display_name or "") != live_name:
        bp.display_name = live_name
        changed = True
    if live_addr and (bp.billing_address or "") != live_addr:
        bp.billing_address = live_addr
        changed = True
    if changed:
        db.session.flush()
        logger.info(
            "[PDF] Snapshot PATIENT aligné sur les données live "
            "(invoice_id=%s, billing_party_id=%s)",
            getattr(invoice, "id", None),
            getattr(bp, "id", None),
        )


def refresh_recipient_snapshot_meta(invoice: Invoice) -> None:
    """Si un recipient_snapshot existe, le remplacer par les valeurs courantes."""
    meta = normalize_invoice_meta_dict(invoice.meta)
    if "recipient_snapshot" not in meta:
        return
    bp = getattr(invoice, "billing_party", None)
    if bp is None:
        return
    prev = meta.get("recipient_snapshot")
    snap = dict(prev) if isinstance(prev, dict) else {}
    snap["billing_party_id"] = bp.id
    snap["display_name"] = bp.display_name
    snap["billing_address"] = bp.billing_address
    snap["contact_email"] = getattr(bp, "contact_email", None)
    snap["contact_phone"] = getattr(bp, "contact_phone", None)
    meta["recipient_snapshot"] = snap
    invoice.meta = meta


def _delete_replaced_invoice_pdf(old_url: str | None, new_url: str | None) -> None:
    """Supprime l'ancien fichier seulement après remplacement réussi."""
    if not old_url or not new_url or old_url == new_url:
        return
    try:
        from shared.upload_path_resolver import (
            get_uploads_base,
            resolve_safe_upload_path,
        )

        path = resolve_safe_upload_path(old_url, uploads_base=get_uploads_base())
        if path.suffix.lower() != ".pdf":
            return
        path.unlink()
        logger.info("[PDF] Ancien fichier remplacé supprimé: %s", old_url)
    except Exception:
        logger.warning(
            "[PDF] Ancien fichier non supprimé après régénération: %s",
            old_url,
            exc_info=True,
        )


class ForceRegenerateInvoicePdfUseCase:
    """Opération unique : reconstruction forcée du PDF depuis la DB courante."""

    def execute(
        self, *, company_id: int, invoice_id: int
    ) -> ForceRegenerateInvoicePdfResult:
        invoice = reload_invoice_graph_for_pdf(invoice_id, company_id)
        if invoice is None:
            return ForceRegenerateInvoicePdfResult(
                ok=False,
                error={"error": "Facture non trouvée"},
                status_code=404,
            )

        if not invoice_allows_line_editing(invoice):
            return ForceRegenerateInvoicePdfResult(
                ok=False,
                previous_pdf_url=invoice.pdf_url,
                error={
                    "error": (
                        f"Impossible de régénérer le PDF: la facture est "
                        f"{invoice.status.value} et ne peut plus être modifiée."
                    )
                },
                status_code=400,
            )

        previous_pdf_url = invoice.pdf_url
        sync_patient_billing_party_from_live(invoice)
        refresh_recipient_snapshot_meta(invoice)

        try:
            from application.invoices.paper_invoice_fee import ensure_paper_invoice_fee_line

            if ensure_paper_invoice_fee_line(
                invoice, client=getattr(invoice, "client", None)
            ):
                db.session.flush()
                invoice = reload_invoice_graph_for_pdf(invoice_id, company_id) or invoice
        except Exception:
            logger.exception(
                "ensure_paper_invoice_fee_line échoué invoice_id=%s", invoice_id
            )

        try:
            pdf_result = GenerateInvoicePdfUseCase().execute(
                invoice=invoice, force_regenerate=True
            )
        except Exception:
            logger.exception("Régénération PDF interrompue invoice_id=%s", invoice_id)
            db.session.rollback()
            return ForceRegenerateInvoicePdfResult(
                ok=False,
                previous_pdf_url=previous_pdf_url,
                error={"error": "Erreur lors de la génération du PDF"},
                status_code=500,
            )

        if pdf_result.ok and pdf_result.pdf_url:
            # Recharger l'identité après expire_all interne à generate_invoice_pdf.
            invoice = Invoice.query.get(invoice_id) or invoice
            mark_pdf_ready(invoice, pdf_result.pdf_url)
            db.session.commit()
            _delete_replaced_invoice_pdf(previous_pdf_url, pdf_result.pdf_url)
            generated_at = None
            meta = normalize_invoice_meta_dict(invoice.meta)
            pdf_meta = meta.get("pdf")
            if isinstance(pdf_meta, dict):
                ga = pdf_meta.get("generated_at")
                generated_at = ga if isinstance(ga, str) else None
            return ForceRegenerateInvoicePdfResult(
                ok=True,
                pdf_url=pdf_result.pdf_url,
                generated_at=generated_at,
                previous_pdf_url=previous_pdf_url,
            )

        # Échec : ne jamais retirer l'ancien PDF ni déclarer le succès.
        db.session.rollback()
        err = (
            pdf_result.error
            if isinstance(pdf_result.error, dict)
            else {"error": "Impossible de régénérer le PDF"}
        )
        logger.error(
            "Régénération PDF échouée invoice_id=%s previous_pdf_url=%s error=%s",
            invoice_id,
            previous_pdf_url,
            err,
        )
        return ForceRegenerateInvoicePdfResult(
            ok=False,
            previous_pdf_url=previous_pdf_url,
            error=err,
            status_code=pdf_result.status_code or 500,
        )


def force_regenerate_invoice_pdf(
    *, company_id: int, invoice_id: int
) -> ForceRegenerateInvoicePdfResult:
    """Point d'entrée unique (HTTP, Celery, autres use cases)."""
    return ForceRegenerateInvoicePdfUseCase().execute(
        company_id=company_id, invoice_id=invoice_id
    )
