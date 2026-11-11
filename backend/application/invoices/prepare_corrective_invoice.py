"""Prépare une facture rectificative liée à l'original et à un avoir.

- Nouveau numéro (séquence mensuelle).
- Ne modifie pas l'original (lignes, PDF, paiements).
- Ne réimpute aucun acompte.
- Ne revendique pas les réservations (``reservation_id`` null, meta de lien).
"""

from __future__ import annotations

import copy
import logging
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import selectinload

from domain.end_of_trip_settlement.cash_settlement import money
from ext import db
from infrastructure.invoices.invoice_number_generator import InvoiceNumberGenerator
from models import (
    CompanyBillingSettings,
    Invoice,
    InvoiceCreditNote,
    InvoiceLine,
    InvoiceSequence,
)
from models.enums import InvoiceLineType, InvoiceStatus

logger = logging.getLogger(__name__)

_CENT = Decimal("0.01")
_ZERO = Decimal("0.00")
_KEY_MAX = 80
_REASON_MAX = 2000


class CorrectiveInvoiceRefusal(Exception):
    """Refus de préparation d'une facture rectificative."""

    def __init__(self, code: str, status: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def _money(value: Any) -> Decimal:
    return money(value)


def _lock_invoice(invoice_id: int, company_id: int) -> Invoice | None:
    return db.session.scalar(
        select(Invoice)
        .where(Invoice.id == int(invoice_id), Invoice.company_id == int(company_id))
        .options(selectinload(Invoice.lines))
        .with_for_update()
    )


def _by_key(company_id: int, key: str) -> Invoice | None:
    return db.session.scalar(
        select(Invoice)
        .where(
            Invoice.company_id == int(company_id),
            Invoice.correction_idempotency_key == key,
        )
        .with_for_update()
    )


def _allocate_number(company_id: int, year: int, month: int) -> str:
    sequence = (
        InvoiceSequence.query.filter_by(
            company_id=int(company_id), year=int(year), month=int(month)
        )
        .with_for_update()
        .first()
    )
    if sequence is None:
        nested = db.session.begin_nested()
        try:
            sequence = InvoiceSequence(
                company_id=int(company_id),
                year=int(year),
                month=int(month),
                sequence=0,
            )
            db.session.add(sequence)
            db.session.flush()
            nested.commit()
        except IntegrityError as err:
            nested.rollback()
            sequence = (
                InvoiceSequence.query.filter_by(
                    company_id=int(company_id), year=int(year), month=int(month)
                )
                .with_for_update()
                .first()
            )
            if sequence is None:
                raise CorrectiveInvoiceRefusal("sequence_indisponible", 500) from err
    sequence.sequence = int(sequence.sequence) + 1
    db.session.flush()
    settings = CompanyBillingSettings.query.filter_by(
        company_id=int(company_id)
    ).first()
    if settings is None:
        prefix = "EM"
        fmt = "{PREFIX}-{YYYY}-{MM}-{SEQ4}"
    else:
        prefix = settings.invoice_prefix or "EM"
        fmt = settings.invoice_number_format or "{PREFIX}-{YYYY}-{MM}-{SEQ4}"
    billing = type(
        "Billing",
        (),
        {"invoice_prefix": prefix, "invoice_number_format": fmt},
    )()
    return InvoiceNumberGenerator().generate(
        company_id,
        year,
        month,
        billing,  # type: ignore[arg-type]
        type(
            "Seq",
            (),
            {
                "id": sequence.id,
                "company_id": company_id,
                "year": year,
                "month": month,
                "sequence": sequence.sequence,
            },
        )(),  # type: ignore[arg-type]
    )


def _payload(draft: Invoice, *, replay: bool) -> dict[str, Any]:
    return {
        "corrective_invoice_id": int(draft.id),
        "invoice_number": draft.invoice_number,
        "status": (
            draft.status.value if hasattr(draft.status, "value") else str(draft.status)
        ),
        "corrects_invoice_id": int(draft.corrects_invoice_id or 0) or None,
        "correction_credit_note_id": int(draft.correction_credit_note_id or 0) or None,
        "correction_reason": draft.correction_reason,
        "total_amount": float(_money(draft.total_amount)),
        "amount_paid": float(_money(draft.amount_paid)),
        "balance_due": float(_money(draft.balance_due)),
        "replay": replay,
        "payments_copied": False,
        "bookings_claimed": False,
    }


def prepare_corrective_invoice(
    *,
    company_id: int,
    invoice_id: int,
    credit_note_id: int,
    reason: str,
    idempotency_key: str,
    created_by_user_id: int,
) -> dict[str, Any]:
    """Crée un brouillon rectificatif lié à l'original et à l'avoir."""
    del created_by_user_id  # traçabilité via correction_reason / meta
    key = (idempotency_key or "").strip()
    if not key:
        raise CorrectiveInvoiceRefusal("cle_idempotence_requise", 400)
    if len(key) > _KEY_MAX:
        raise CorrectiveInvoiceRefusal("cle_idempotence_trop_longue", 400)
    motive = (reason or "").strip()
    if not motive:
        raise CorrectiveInvoiceRefusal("motif_requis", 400)
    if len(motive) > _REASON_MAX:
        raise CorrectiveInvoiceRefusal("motif_trop_long", 400)

    original = _lock_invoice(invoice_id, company_id)
    if original is None:
        raise CorrectiveInvoiceRefusal("facture_absente", 404)
    status = str(getattr(original.status, "value", original.status) or "").lower()
    if status == InvoiceStatus.DRAFT.value:
        raise CorrectiveInvoiceRefusal("brouillon_refuse")
    if status == InvoiceStatus.CANCELLED.value:
        raise CorrectiveInvoiceRefusal("facture_annulee")

    existing = _by_key(company_id, key)
    if existing is not None:
        if (
            int(existing.corrects_invoice_id or 0) != int(original.id)
            or int(existing.correction_credit_note_id or 0) != int(credit_note_id)
            or (existing.correction_reason or "") != motive
        ):
            raise CorrectiveInvoiceRefusal("conflit_idempotence")
        return _payload(existing, replay=True)

    note = db.session.scalar(
        select(InvoiceCreditNote)
        .where(
            InvoiceCreditNote.id == int(credit_note_id),
            InvoiceCreditNote.invoice_id == int(original.id),
            InvoiceCreditNote.company_id == int(company_id),
        )
        .options(selectinload(InvoiceCreditNote.line_allocations))
        .with_for_update()
    )
    if note is None:
        raise CorrectiveInvoiceRefusal("avoir_introuvable", 404)
    allocations = list(note.line_allocations or [])
    if not allocations:
        raise CorrectiveInvoiceRefusal("lignes_requises_pour_rectificative", 400)

    source_lines = {int(line.id): line for line in (original.lines or [])}
    now = datetime.now(UTC)
    year, month = now.year, now.month
    number = _allocate_number(company_id, year, month)

    meta = copy.deepcopy(original.meta) if isinstance(original.meta, dict) else {}
    meta = dict(meta or {})
    # Ne pas hériter de l'état documentaire / livraison de l'original :
    # sinon ensure_draft_pdf_ready_for_send croit le PDF prêt sans fichier.
    for stale_key in (
        "pdf",
        "specimen",
        "payable",
        "email_deliveries",
        "last_recipient_email",
        "last_email_sent_at",
        "delivery_method",
    ):
        meta.pop(stale_key, None)
    meta["correction"] = {
        "corrects_invoice_id": int(original.id),
        "corrects_invoice_number": original.invoice_number,
        "credit_note_id": int(note.id),
        "credit_number": note.credit_number,
        "prepared_at": now.isoformat(),
        "bookings_claimed": False,
        "payments_copied": False,
    }

    draft = Invoice(
        company_id=int(company_id),
        client_id=int(original.client_id),
        bill_to_client_id=original.bill_to_client_id,
        billing_party_id=original.billing_party_id,
        billing_strategy=original.billing_strategy,
        billed_to_company_id=original.billed_to_company_id,
        institution_patient_id=original.institution_patient_id,
        period_month=int(month),
        period_year=int(year),
        invoice_number=number,
        currency=original.currency or "CHF",
        subtotal_amount=_ZERO,
        late_fee_amount=_ZERO,
        reminder_fee_amount=_ZERO,
        vat_total_amount=_ZERO,
        vat_breakdown=None,
        total_amount=_ZERO,
        amount_paid=_ZERO,
        balance_due=_ZERO,
        refundable_chf=_ZERO,
        issued_at=now,
        due_date=now + timedelta(days=30),
        status=InvoiceStatus.DRAFT,
        pdf_url=None,
        qr_reference=None,
        meta=meta,
        corrects_invoice_id=int(original.id),
        correction_credit_note_id=int(note.id),
        correction_reason=motive,
        correction_idempotency_key=key,
    )
    nested = db.session.begin_nested()
    db.session.add(draft)
    try:
        db.session.flush()
    except IntegrityError as exc:
        nested.rollback()
        raced = _by_key(company_id, key)
        if raced is None:
            raise CorrectiveInvoiceRefusal("conflit_idempotence") from exc
        return _payload(raced, replay=True)

    subtotal = _ZERO
    vat_total = _ZERO
    total = _ZERO
    vat_map: dict[str, Decimal] = {}

    for alloc in allocations:
        source = source_lines.get(int(alloc.invoice_line_id))
        if source is None:
            raise CorrectiveInvoiceRefusal("ligne_source_absente", 400)
        ht = _money(alloc.amount_ht)
        vat = _money(alloc.vat_amount)
        ttc = _money(alloc.amount_ttc)
        rate = alloc.vat_rate if alloc.vat_rate is not None else source.vat_rate
        src_meta = dict(source.line_meta) if isinstance(source.line_meta, dict) else {}
        for claim_key in (
            "booking_ids",
            "reservation_ids",
            "round_trip_secondary_reservation_ids",
            "round_trip_secondary_reservation_id",
            "round_trip_merge_partner_reservation_id",
            "round_trip_merge_primary_reservation_id",
        ):
            src_meta.pop(claim_key, None)
        src_meta["corrects_line_id"] = int(source.id)
        src_meta["original_reservation_id"] = source.reservation_id
        src_meta["correction_credit_note_id"] = int(note.id)
        src_meta["do_not_claim_booking"] = True
        line_type = source.type
        if not isinstance(line_type, InvoiceLineType):
            try:
                line_type = InvoiceLineType(str(line_type))
            except Exception:
                line_type = InvoiceLineType.CUSTOM
        db.session.add(
            InvoiceLine(
                invoice_id=int(draft.id),
                type=line_type,
                description=str(source.description or "")[:500],
                qty=_money(source.qty) or Decimal("1.00"),
                unit_price=ht,
                line_total=ht,
                vat_rate=rate,
                vat_amount=vat,
                total_with_vat=ttc,
                adjustment_note=(
                    f"Correction de la ligne {source.id} "
                    f"(facture {original.invoice_number})"
                ),
                reservation_id=None,
                line_meta=src_meta,
            )
        )
        subtotal += ht
        vat_total += vat
        total += ttc
        rate_key = f"{float(rate):.2f}" if rate is not None else "0.00"
        vat_map[rate_key] = vat_map.get(rate_key, _ZERO) + vat

    draft.subtotal_amount = subtotal.quantize(_CENT)
    draft.vat_total_amount = vat_total.quantize(_CENT)
    draft.total_amount = total.quantize(_CENT)
    draft.balance_due = total.quantize(_CENT)
    draft.amount_paid = _ZERO
    draft.vat_breakdown = {
        "by_rate": {k: float(v.quantize(_CENT)) for k, v in vat_map.items()}
    }
    nested.commit()
    db.session.commit()
    fresh = db.session.get(Invoice, draft.id)
    if fresh is None:
        raise CorrectiveInvoiceRefusal("rectificative_introuvable", 404)
    logger.info(
        "Rectificative préparée draft_id=%s corrects=%s credit_note=%s",
        fresh.id,
        original.id,
        note.id,
    )
    return _payload(fresh, replay=False)
