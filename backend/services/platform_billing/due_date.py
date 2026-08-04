"""Modification / prolongation d'échéance des factures plateforme."""

from __future__ import annotations

from datetime import UTC, datetime

from ext import db
from models.enums import PlatformDueDateChangeType, PlatformIssuedDocumentType
from models.platform_billing import (
    PlatformInvoiceDueDateChange,
    PlatformIssuedInvoice,
)
from services.platform_billing.payments import (
    _lock_invoice,
    recompute_invoice_payment_state,
)


def update_issued_invoice_due_date(
    issued_invoice_id: int,
    *,
    new_due_at: datetime,
    reason: str,
    admin_user_id: int | None = None,
) -> PlatformIssuedInvoice:
    """Corrige (avant envoi) ou prolonge (après envoi) l'échéance.

    Après envoi : PDF immuable ; réconciliation dunning obligatoire.
    Avant envoi : régénération PDF autorisée.
    """
    inv = _lock_invoice(issued_invoice_id)
    reason_clean = (reason or "").strip()
    if not reason_clean:
        raise ValueError("Motif de changement d'échéance obligatoire")
    if getattr(inv, "document_type", None) == PlatformIssuedDocumentType.CREDIT_NOTE.value:
        raise ValueError("Échéance non modifiable sur une note de crédit")
    if inv.status in ("CANCELLED", "CREDITED", "DRAFT"):
        raise ValueError("Échéance non modifiable pour ce statut")

    new_due = new_due_at if new_due_at.tzinfo else new_due_at.replace(tzinfo=UTC)
    issued = inv.issued_at
    if issued is not None:
        issued_aware = issued if issued.tzinfo else issued.replace(tzinfo=UTC)
        if new_due < issued_aware:
            raise ValueError("L'échéance ne peut pas être antérieure à la date d'émission")

    old_due = inv.due_at
    old_checksum = inv.pdf_checksum
    sent = inv.sent_at is not None

    if sent:
        if old_due is None:
            raise ValueError("Échéance d'origine absente")
        old_aware = old_due if old_due.tzinfo else old_due.replace(tzinfo=UTC)
        if new_due < old_aware:
            raise ValueError(
                "Après envoi, seule une prolongation d'échéance est autorisée"
            )
        change_type = PlatformDueDateChangeType.EXTENSION_AFTER_SEND.value
    else:
        change_type = PlatformDueDateChangeType.CORRECTION_BEFORE_SEND.value

    inv.due_at = new_due
    new_checksum = old_checksum

    if not sent:
        # Régénération PDF avant envoi
        from services.platform_billing.issuance import regenerate_issued_invoice_pdf

        regenerate_issued_invoice_pdf(inv.id, commit=False)
        db.session.refresh(inv)
        new_checksum = inv.pdf_checksum

    db.session.add(
        PlatformInvoiceDueDateChange(
            issued_invoice_id=inv.id,
            old_due_at=old_due,
            new_due_at=new_due,
            reason=reason_clean[:512],
            change_type=change_type,
            admin_user_id=admin_user_id,
            old_pdf_checksum=old_checksum,
            new_pdf_checksum=new_checksum,
        )
    )
    recompute_invoice_payment_state(inv)

    if sent:
        from services.platform_billing.dunning import (
            reconcile_invoice_dunning_after_due_date_change,
        )

        reconcile_invoice_dunning_after_due_date_change(inv.id)

    db.session.commit()
    db.session.refresh(inv)
    return inv
