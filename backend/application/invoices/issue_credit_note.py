"""Émission d'un avoir sur une facture patient déjà émise.

L'avoir réduit la créance. Il n'écrit aucun paiement et aucun remboursement.
Le PDF d'origine n'est pas régénéré. Le PDF d'avoir est un autre fichier,
écrit après le commit de la ligne d'avoir.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from application.invoices.credit_note_line_math import preview_credit_for_lines
from domain.end_of_trip_settlement.cash_settlement import money
from domain.end_of_trip_settlement.financial_contract import credit_room
from ext import db
from models import Invoice, InvoiceCreditNote, InvoiceCreditSequence
from models.enums import InvoiceStatus
from models.invoice import InvoiceCreditNoteLine
from services.documents.credit_note_pdf import CreditNoteDocument, write_credit_note_pdf

logger = logging.getLogger(__name__)

_CENT = Decimal("0.01")
_KEY_MAX = 80
_REASON_MAX = 2000


class CreditNoteRefusal(Exception):
    """Refus d'émission d'un avoir."""

    def __init__(self, code: str, status: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.status = status


def _money(value: Any) -> Decimal:
    return money(value)


def _status(invoice: Invoice) -> str:
    return str(getattr(invoice.status, "value", invoice.status) or "").lower()


def _lock_invoice(invoice_id: int, company_id: int) -> Invoice | None:
    return db.session.scalar(
        select(Invoice)
        .where(Invoice.id == int(invoice_id), Invoice.company_id == int(company_id))
        .with_for_update()
    )


def _note_by_key(company_id: int, key: str) -> InvoiceCreditNote | None:
    return db.session.scalar(
        select(InvoiceCreditNote)
        .where(
            InvoiceCreditNote.company_id == int(company_id),
            InvoiceCreditNote.idempotency_key == key,
        )
        .with_for_update()
    )


def _same_request(
    note: InvoiceCreditNote,
    invoice_id: int,
    amount: Decimal,
    reason: str,
    line_specs: list[dict[str, Any]] | None,
) -> bool:
    if (
        int(note.invoice_id) != int(invoice_id)
        or _money(note.amount_chf) != amount
        or note.reason != reason
    ):
        return False
    existing = [
        {
            "invoice_line_id": int(row.invoice_line_id),
            "amount_ttc": _money(row.amount_ttc),
        }
        for row in (note.line_allocations or [])
    ]
    existing.sort(key=lambda row: row["invoice_line_id"])
    expected = [
        {
            "invoice_line_id": int(row["invoice_line_id"]),
            "amount_ttc": _money(row["amount_ttc"]),
        }
        for row in (line_specs or [])
    ]
    expected.sort(key=lambda row: row["invoice_line_id"])
    return existing == expected


def _next_number(company_id: int, year: int) -> str:
    row = db.session.scalar(
        select(InvoiceCreditSequence)
        .where(
            InvoiceCreditSequence.company_id == int(company_id),
            InvoiceCreditSequence.year == int(year),
        )
        .with_for_update()
    )
    if row is None:
        nested = db.session.begin_nested()
        try:
            row = InvoiceCreditSequence(
                company_id=int(company_id), year=int(year), last_value=0
            )
            db.session.add(row)
            db.session.flush()
            nested.commit()
        except IntegrityError:
            nested.rollback()
            row = db.session.scalar(
                select(InvoiceCreditSequence)
                .where(
                    InvoiceCreditSequence.company_id == int(company_id),
                    InvoiceCreditSequence.year == int(year),
                )
                .with_for_update()
            )
            if row is None:
                raise
    row.last_value = int(row.last_value) + 1
    return f"AV-{int(company_id)}-{int(year)}-{int(row.last_value)}"


def _active_amounts(invoice_id: int, company_id: int) -> list[Decimal]:
    rows = InvoiceCreditNote.query.filter_by(
        invoice_id=int(invoice_id), company_id=int(company_id)
    ).all()
    return [_money(row.amount_chf) for row in rows]


def _day(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).date().isoformat()


def _party(invoice: Invoice) -> tuple[str, str]:
    party = getattr(invoice, "billing_party", None)
    if party is not None and getattr(party, "display_name", None):
        return str(party.display_name), str(party.billing_address or "")
    client = getattr(invoice, "client", None)
    user = getattr(client, "user", None) if client is not None else None
    name = " ".join(
        part
        for part in (
            getattr(user, "first_name", None),
            getattr(user, "last_name", None),
        )
        if part
    ).strip()
    return name or "Destinataire", ""


def _document(invoice: Invoice, note: InvoiceCreditNote) -> CreditNoteDocument:
    company = invoice.company
    recipient_name, recipient_address = _party(invoice)
    issued = note.issued_at or datetime.now(UTC)
    invoice_issued = invoice.issued_at or issued
    if note.line_allocations:
        line_by_id = {int(line.id): line for line in (invoice.lines or [])}
        lines = tuple(
            (
                str(
                    getattr(
                        line_by_id.get(int(alloc.invoice_line_id)), "description", None
                    )
                    or f"Ligne {alloc.invoice_line_id}"
                ),
                f"{_money(alloc.amount_ttc):.2f}",
            )
            for alloc in note.line_allocations
        )
    else:
        lines = tuple(
            (str(line.description), f"{_money(line.line_total):.2f}")
            for line in invoice.lines
        )
    note_amounts = _active_amounts(invoice.id, invoice.company_id)
    credit_total = sum(note_amounts, Decimal("0.00")).quantize(_CENT)
    original = _money(invoice.total_amount)
    net = (original - credit_total).quantize(_CENT)
    return CreditNoteDocument(
        credit_number=note.credit_number,
        issued_on=_day(issued),
        amount_chf=f"{_money(note.amount_chf):.2f}",
        reason=note.reason,
        invoice_number=str(invoice.invoice_number),
        invoice_issued_on=_day(invoice_issued),
        company_name=str(getattr(company, "name", "") or "Entreprise"),
        company_ide=str(getattr(company, "uid_ide", "") or ""),
        recipient_name=recipient_name,
        recipient_address=recipient_address or " ",
        lines=lines,
        original_total=f"{original:.2f}",
        credit_total=f"{credit_total:.2f}",
        net_invoiced=f"{net:.2f}",
        amount_paid=f"{_money(invoice.amount_paid):.2f}",
        balance_due=f"{_money(invoice.balance_due):.2f}",
        overpayment=f"{_money(invoice.refundable_chf):.2f}",
    )


def _payload(
    invoice: Invoice,
    note: InvoiceCreditNote,
    *,
    replay: bool,
    pdf_pending: bool,
) -> dict[str, Any]:
    note_amounts = _active_amounts(invoice.id, invoice.company_id)
    credit_total = sum(note_amounts, Decimal("0.00")).quantize(_CENT)
    original = _money(invoice.total_amount)
    return {
        "credit_note_id": int(note.id),
        "credit_number": note.credit_number,
        "amount_chf": float(_money(note.amount_chf)),
        "reason": note.reason,
        "invoice_id": int(invoice.id),
        "invoice_number": invoice.invoice_number,
        "invoice_pdf_url": invoice.pdf_url,
        "pdf_url": note.pdf_url,
        "pdf_pending": pdf_pending,
        "replay": replay,
        "original_total": float(original),
        "credit_notes_total": float(credit_total),
        "net_invoiced_total": float((original - credit_total).quantize(_CENT)),
        "amount_paid": float(_money(invoice.amount_paid)),
        "balance_due": float(_money(invoice.balance_due)),
        "overpayment_amount": float(_money(invoice.refundable_chf)),
        "status": _status(invoice),
        "lines": [
            {
                "invoice_line_id": int(alloc.invoice_line_id),
                "amount_ht": float(_money(alloc.amount_ht)),
                "vat_amount": float(_money(alloc.vat_amount)),
                "amount_ttc": float(_money(alloc.amount_ttc)),
                "vat_rate": (
                    float(alloc.vat_rate) if alloc.vat_rate is not None else None
                ),
            }
            for alloc in (note.line_allocations or [])
        ],
    }


def _ensure_pdf(
    invoice: Invoice, note: InvoiceCreditNote, *, replay: bool
) -> dict[str, Any]:
    if note.pdf_url:
        return _payload(invoice, note, replay=replay, pdf_pending=False)
    try:
        url = write_credit_note_pdf(_document(invoice, note))
    except Exception:
        logger.exception("PDF d'avoir non écrit credit_number=%s", note.credit_number)
        return _payload(invoice, note, replay=replay, pdf_pending=True)
    locked = db.session.scalar(
        select(InvoiceCreditNote)
        .where(InvoiceCreditNote.id == int(note.id))
        .with_for_update()
    )
    if locked is None:
        raise CreditNoteRefusal("avoir_introuvable", 404)
    if locked.pdf_url is None:
        locked.pdf_url = url
    db.session.commit()
    db.session.refresh(invoice)
    return _payload(invoice, locked, replay=replay, pdf_pending=False)


def _resolve_amount_and_lines(
    invoice: Invoice,
    *,
    amount_chf: Decimal | None,
    line_selections: list[dict[str, Any]] | None,
) -> tuple[Decimal, list[dict[str, Any]] | None]:
    """Le serveur calcule le montant si des lignes sont fournies."""
    if line_selections:
        try:
            preview = preview_credit_for_lines(invoice, line_selections)
        except ValueError as exc:
            code = str(exc)
            mapping = {
                "avoir_au_dela_du_facture": "avoir_au_dela_du_facture",
                "avoir_au_dela_de_la_ligne": "avoir_au_dela_de_la_ligne",
                "ligne_inconnue": "ligne_inconnue",
                "ligne_en_double": "ligne_en_double",
                "ligne_deja_creditee": "ligne_deja_creditee",
                "lignes_requises": "lignes_requises",
                "ligne_sans_montant": "montant_invalide",
                "montant_invalide": "montant_invalide",
            }
            raise CreditNoteRefusal(mapping.get(code, "montant_invalide"), 400) from exc
        amount = _money(preview["proposed_amount_ttc"])
        if amount_chf is not None and _money(amount_chf) != amount:
            # Le client peut renvoyer le montant serveur ; tout autre écart est refusé.
            raise CreditNoteRefusal("montant_incoherent_avec_lignes", 400)
        return amount, list(preview["lines"])
    if amount_chf is None:
        raise CreditNoteRefusal("montant_invalide", 400)
    amount = _money(amount_chf)
    if amount <= 0:
        raise CreditNoteRefusal("montant_invalide", 400)
    return amount, None


def issue_credit_note(
    *,
    company_id: int,
    invoice_id: int,
    reason: str,
    idempotency_key: str,
    created_by_user_id: int,
    amount_chf: Decimal | None = None,
    lines: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Émet un avoir, ou rejoue la même demande.

    Si ``lines`` est fourni, le montant TTC et la TVA sont calculés côté serveur.
    """
    key = (idempotency_key or "").strip()
    if not key:
        raise CreditNoteRefusal("cle_idempotence_requise", 400)
    if len(key) > _KEY_MAX:
        raise CreditNoteRefusal("cle_idempotence_trop_longue", 400)
    motive = (reason or "").strip()
    if not motive:
        raise CreditNoteRefusal("motif_requis", 400)
    if len(motive) > _REASON_MAX:
        raise CreditNoteRefusal("motif_trop_long", 400)

    invoice = _lock_invoice(invoice_id, company_id)
    if invoice is None:
        raise CreditNoteRefusal("facture_absente", 404)
    db.session.refresh(invoice)
    status = _status(invoice)
    if status == InvoiceStatus.DRAFT.value:
        raise CreditNoteRefusal("brouillon_refuse")
    if status == InvoiceStatus.CANCELLED.value:
        raise CreditNoteRefusal("facture_annulee")
    if status not in {
        InvoiceStatus.SENT.value,
        InvoiceStatus.PARTIALLY_PAID.value,
        InvoiceStatus.PAID.value,
        InvoiceStatus.OVERDUE.value,
    }:
        raise CreditNoteRefusal("facture_non_emise")

    try:
        amount, line_specs = _resolve_amount_and_lines(
            invoice, amount_chf=amount_chf, line_selections=lines
        )
    except CreditNoteRefusal:
        raise
    except ValueError as exc:
        raise CreditNoteRefusal("montant_invalide", 400) from exc

    existing = _note_by_key(company_id, key)
    if existing is not None:
        if not _same_request(existing, invoice.id, amount, motive, line_specs):
            raise CreditNoteRefusal("conflit_idempotence")
        note_id = int(existing.id)
        db.session.rollback()
        reloaded = db.session.get(Invoice, invoice_id)
        replayed = db.session.get(InvoiceCreditNote, note_id)
        if reloaded is None or replayed is None:
            raise CreditNoteRefusal("facture_absente", 404)
        return _ensure_pdf(reloaded, replayed, replay=True)

    try:
        credit_room(
            invoice.total_amount, [*_active_amounts(invoice.id, company_id), amount]
        )
    except ValueError as exc:
        code = str(exc)
        if code == "avoir_au_dela_du_facture":
            raise CreditNoteRefusal("avoir_au_dela_du_facture") from exc
        raise CreditNoteRefusal("montant_invalide", 400) from exc

    year = datetime.now(UTC).year
    number = _next_number(company_id, year)
    note = InvoiceCreditNote(
        company_id=int(company_id),
        invoice_id=int(invoice.id),
        credit_number=number,
        amount_chf=amount,
        reason=motive,
        status="issued",
        issued_at=datetime.now(UTC),
        created_by_user_id=int(created_by_user_id),
        idempotency_key=key,
        pdf_url=None,
    )
    nested = db.session.begin_nested()
    db.session.add(note)
    try:
        db.session.flush()
        if line_specs:
            for spec in line_specs:
                db.session.add(
                    InvoiceCreditNoteLine(
                        credit_note_id=int(note.id),
                        invoice_line_id=int(spec["invoice_line_id"]),
                        amount_ht=_money(spec["amount_ht"]),
                        vat_amount=_money(spec["vat_amount"]),
                        amount_ttc=_money(spec["amount_ttc"]),
                        vat_rate=(
                            _money(spec["vat_rate"])
                            if spec.get("vat_rate") is not None
                            else None
                        ),
                    )
                )
            db.session.flush()
    except IntegrityError as exc:
        nested.rollback()
        raced = _note_by_key(company_id, key)
        if raced is None or not _same_request(
            raced, invoice.id, amount, motive, line_specs
        ):
            raise CreditNoteRefusal("conflit_idempotence") from exc
        raced_id = int(raced.id)
        db.session.rollback()
        reloaded = db.session.get(Invoice, invoice_id)
        replayed = db.session.get(InvoiceCreditNote, raced_id)
        if reloaded is None or replayed is None:
            raise CreditNoteRefusal("facture_absente", 404) from exc
        return _ensure_pdf(reloaded, replayed, replay=True)
    else:
        nested.commit()

    invoice.update_balance()
    db.session.commit()
    stored = db.session.get(InvoiceCreditNote, note.id)
    fresh = db.session.get(Invoice, invoice.id)
    if stored is None or fresh is None:
        raise CreditNoteRefusal("avoir_introuvable", 404)
    return _ensure_pdf(fresh, stored, replay=False)
