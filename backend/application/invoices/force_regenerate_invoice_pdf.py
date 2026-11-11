"""Régénération FORCÉE du PDF facture depuis l'état courant de la base.

Contrat figé (CLOSED) pour tous les points d'entrée (ligne de facture, modale
brouillon, tâche Celery) : relire la facture et ses relations, ignorer l'ancien
binaire / snapshot périmé, écrire un nouveau fichier, remplacer la référence
seulement après succès. En cas d'échec, l'ancien PDF reste servi.

Étape 22 (LIRIE) — ce contrat ne s'applique qu'aux **brouillons**. Une facture
émise (tout statut autre que ``draft``) est refusée **avant** toute écriture en
base, tout recalcul et toute suppression de fichier : son PDF original, ses
lignes, ses totaux, sa TVA, ses références QR et ses snapshots restent
intacts (409 ``ISSUED_INVOICE_IMMUTABLE``). Le statut faisant autorité est lu
sous verrou de ligne (``SELECT … FOR UPDATE``) afin de sérialiser deux
régénérations concurrentes et une régénération concurrente d'un changement de
statut.

Voir docs/facturation/regenerer-pdf-contrat.md et
docs/contracts/encaissement-audit/etape-22/.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import joinedload, selectinload

from application.invoices.generate_invoice_pdf import GenerateInvoicePdfUseCase
from application.invoices.invoice_pdf_state import (
    mark_pdf_ready,
    normalize_invoice_meta_dict,
)
from application.invoices.issued_invoice_guard import (
    IssuedInvoiceImmutableError,
    assert_invoice_mutable,
    invoice_is_issued,
)
from application.invoices.paper_invoice_fee import ensure_paper_invoice_fee_line
from ext import db
from models import Client, Invoice

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ForceRegenerateInvoicePdfResult:
    ok: bool
    pdf_url: str | None = None
    generated_at: str | None = None
    previous_pdf_url: str | None = None
    error: dict[str, Any] | None = None
    status_code: int | None = None


@dataclass(frozen=True, slots=True)
class _LockedInvoiceView:
    """Projection minimale (id, statut, pdf_url) lue sous verrou de ligne."""

    id: int
    status: object
    pdf_url: str | None


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


def reconcile_draft_invoice_billing_party(invoice: Invoice) -> bool:
    """Brouillon : remplace un BP PATIENT technique par le payeur effectif.

    Ne change ni les montants, ni les lignes, ni une facture déjà figée.
    ``sync_patient_billing_party_from_live`` ne fait pas cette bascule : il
    ne réécrit que le nom et l'adresse du BP PATIENT.
    """
    from models import Booking
    from models.billing_party import BillingParty
    from services.billing.billing_party_linker import is_establishment_billing_party
    from services.billing.client_stay_resolver import (
        resolve_default_billing_party_for_client,
    )
    from services.billing.effective_patient_payer import (
        is_technical_patient_billing_party,
        resolve_effective_party_id_for_bookings,
    )
    from services.documents.invoice_recipient import (
        BILLED_TO_SNAPSHOT_KEY,
        invoice_billed_to_is_frozen,
    )

    if invoice_billed_to_is_frozen(invoice):
        return False
    party = getattr(invoice, "billing_party", None)
    if party is None and getattr(invoice, "billing_party_id", None) is not None:
        party = db.session.get(BillingParty, int(invoice.billing_party_id))
    if not is_technical_patient_billing_party(party):
        return False

    company_id = int(invoice.company_id)
    line_ids = [
        int(line.id)
        for line in (getattr(invoice, "lines", None) or [])
        if getattr(line, "id", None) is not None
    ]
    bookings = (
        Booking.query.filter(Booking.invoice_line_id.in_(line_ids)).all()
        if line_ids
        else []
    )
    effective_id: int | None
    if bookings:
        effective_id = resolve_effective_party_id_for_bookings(
            bookings, company_id=company_id
        )
    else:
        third = None
        if getattr(invoice, "client_id", None) is not None:
            third = resolve_default_billing_party_for_client(
                client_id=int(invoice.client_id),
                company_id=company_id,
            )
        if third is not None and not is_establishment_billing_party(third):
            effective_id = int(third.id)
        else:
            effective_id = None
    if effective_id is None or int(effective_id) == int(party.id):
        return False
    target = db.session.get(BillingParty, int(effective_id))
    if target is None or is_technical_patient_billing_party(target):
        return False

    invoice.billing_party_id = int(target.id)
    invoice.billing_party = target
    meta = normalize_invoice_meta_dict(invoice.meta)
    # Le brouillon se rend depuis les master data ; l'ancien snapshot patient_self
    # ne doit plus être servi par un client qui lit ``meta``.
    meta.pop(BILLED_TO_SNAPSHOT_KEY, None)
    if "recipient_snapshot" in meta:
        snap = meta.get("recipient_snapshot")
        snap = dict(snap) if isinstance(snap, dict) else {}
        snap["billing_party_id"] = target.id
        snap["display_name"] = target.display_name
        snap["billing_address"] = target.billing_address
        snap["type"] = getattr(target.type, "value", None) or str(target.type)
        meta["recipient_snapshot"] = snap
    invoice.meta = meta
    db.session.flush()
    logger.info(
        "[PDF] Brouillon réconcilié vers le tiers payeur "
        "(invoice_id=%s, billing_party_id=%s)",
        getattr(invoice, "id", None),
        target.id,
    )
    return True


def sync_patient_billing_party_from_live(invoice: Invoice) -> None:
    """Aligne un BillingParty PATIENT sur le client / patient actuel (brouillon seulement).

    Ne touche pas aux numéros, montants, lignes ni au type de payeur.
    Un tiers payeur (organisme) n'est jamais réécrit depuis le client.
    Une facture figée (hors DRAFT) rend son bloc « Facturé à » depuis le snapshot :
    ses master data ne sont jamais réécrites à l'occasion d'une régénération.
    """
    from models.enums import BillingPartyType
    from services.documents.invoice_recipient import (
        invoice_billed_to_is_frozen,
        live_patient_payer_identity,
    )

    bp = getattr(invoice, "billing_party", None)
    if bp is None:
        return
    if invoice_billed_to_is_frozen(invoice):
        return
    clinic_ref = str(getattr(bp, "external_ref", "") or "")
    if clinic_ref.startswith("clinic_company:"):
        # Parti auto de la clinique : le brouillon suit le nom et l'adresse courants.
        # Un parti mappé à la main (autre external_ref) n'est pas réécrit.
        from services.billing.billing_party_linker import (
            get_or_create_billing_party_for_clinic_company,
        )

        try:
            clinic_company_id = int(clinic_ref.split(":", 1)[1])
        except ValueError:
            return
        get_or_create_billing_party_for_clinic_company(
            company_id=int(invoice.company_id),
            clinic_company_id=clinic_company_id,
        )
        db.session.refresh(bp)
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
    """Brouillon : si un recipient_snapshot (payeur) existe, l'aligner sur les valeurs courantes.

    Facture figée : le snapshot payeur n'est plus réécrit (cohérence avec le
    ``billed_to_snapshot`` immuable).
    """
    from services.documents.invoice_recipient import invoice_billed_to_is_frozen

    if invoice_billed_to_is_frozen(invoice):
        return
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


def lock_invoice_row_for_regeneration(
    invoice_id: int, company_id: int
) -> tuple[int, object, str | None] | None:
    """Verrouille la ligne facture et renvoie ``(id, status, pdf_url)`` faisant autorité.

    ``SELECT … FOR UPDATE`` sérialise deux régénérations concurrentes et attend
    la fin d'un changement de statut en cours (envoi, encaissement) : le statut
    relu après l'obtention du verrou est celui qui décide du refus.
    """
    row = db.session.execute(
        select(Invoice.id, Invoice.status, Invoice.pdf_url)
        .where(Invoice.id == invoice_id, Invoice.company_id == company_id)
        .with_for_update()
    ).first()
    if row is None:
        return None
    return int(row[0]), row[1], row[2]


class ForceRegenerateInvoicePdfUseCase:
    """Opération unique : reconstruction forcée du PDF depuis la DB courante.

    Brouillon uniquement. Une facture émise est refusée avant toute mutation.
    """

    def execute(
        self, *, company_id: int, invoice_id: int
    ) -> ForceRegenerateInvoicePdfResult:
        # 1) Statut faisant autorité, sous verrou de ligne, AVANT toute écriture.
        db.session.expire_all()
        locked = lock_invoice_row_for_regeneration(invoice_id, company_id)
        if locked is None:
            db.session.rollback()
            return ForceRegenerateInvoicePdfResult(
                ok=False,
                error={"error": "Facture non trouvée"},
                status_code=404,
            )
        _locked_id, locked_status, locked_pdf_url = locked
        locked_view = _LockedInvoiceView(
            id=_locked_id, status=locked_status, pdf_url=locked_pdf_url
        )
        try:
            assert_invoice_mutable(locked_view, "force_regenerate_invoice_pdf")
        except IssuedInvoiceImmutableError as guard_err:
            # Refus : aucune écriture n'a eu lieu ; on libère le verrou.
            db.session.rollback()
            return ForceRegenerateInvoicePdfResult(
                ok=False,
                previous_pdf_url=locked_pdf_url,
                error=guard_err.to_error_payload(),
                status_code=guard_err.status_code,
            )

        invoice = reload_invoice_graph_for_pdf(invoice_id, company_id)
        if invoice is None:
            db.session.rollback()
            return ForceRegenerateInvoicePdfResult(
                ok=False,
                error={"error": "Facture non trouvée"},
                status_code=404,
            )
        # Défense en profondeur : l'objet rechargé doit lui aussi être un brouillon.
        if invoice_is_issued(invoice):
            db.session.rollback()
            err = IssuedInvoiceImmutableError(
                invoice_id=invoice.id,
                status=getattr(invoice.status, "value", str(invoice.status)),
                operation="force_regenerate_invoice_pdf",
            )
            return ForceRegenerateInvoicePdfResult(
                ok=False,
                previous_pdf_url=invoice.pdf_url,
                error=err.to_error_payload(),
                status_code=err.status_code,
            )

        previous_pdf_url = invoice.pdf_url
        reconcile_draft_invoice_billing_party(invoice)
        db.session.flush()
        sync_patient_billing_party_from_live(invoice)
        refresh_recipient_snapshot_meta(invoice)

        try:
            if ensure_paper_invoice_fee_line(
                invoice, client=getattr(invoice, "client", None)
            ):
                db.session.flush()
                invoice = (
                    reload_invoice_graph_for_pdf(invoice_id, company_id) or invoice
                )
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
            if invoice_is_issued(invoice):
                # La facture a été émise pendant la génération : on ne remplace
                # ni la référence ni le fichier original (aucune suppression).
                db.session.rollback()
                err = IssuedInvoiceImmutableError(
                    invoice_id=invoice.id,
                    status=getattr(invoice.status, "value", str(invoice.status)),
                    operation="force_regenerate_invoice_pdf",
                )
                return ForceRegenerateInvoicePdfResult(
                    ok=False,
                    previous_pdf_url=previous_pdf_url,
                    error=err.to_error_payload(),
                    status_code=err.status_code,
                )
            mark_pdf_ready(invoice, pdf_result.pdf_url)
            db.session.commit()
            # Suppression de l'ancien fichier : chemin brouillon uniquement, après commit.
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
