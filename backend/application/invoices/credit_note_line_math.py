"""Calcul serveur des avoirs par ligne de facture (HT / TVA / TTC).

React ne détermine pas le montant : le serveur calcule le résiduel et la
ventilation TVA. Les avoirs sans répartition ligne restent au niveau facture.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import func, select

from domain.end_of_trip_settlement.cash_settlement import money
from domain.end_of_trip_settlement.financial_contract import credit_room
from ext import db
from models import Invoice, InvoiceCreditNote, InvoiceLine
from models.invoice import InvoiceCreditNoteLine

_CENT = Decimal("0.01")
_ZERO = Decimal("0.00")


def _money(value: Any) -> Decimal:
    return money(value)


def credited_ttc_by_line(invoice_id: int) -> dict[int, Decimal]:
    """Somme TTC déjà créditée par ``invoice_line_id``."""
    rows = db.session.execute(
        select(
            InvoiceCreditNoteLine.invoice_line_id,
            func.coalesce(func.sum(InvoiceCreditNoteLine.amount_ttc), 0),
        )
        .join(
            InvoiceCreditNote,
            InvoiceCreditNote.id == InvoiceCreditNoteLine.credit_note_id,
        )
        .where(InvoiceCreditNote.invoice_id == int(invoice_id))
        .group_by(InvoiceCreditNoteLine.invoice_line_id)
    ).all()
    return {int(line_id): _money(total) for line_id, total in rows}


def invoice_credit_total(invoice_id: int, company_id: int) -> Decimal:
    rows = InvoiceCreditNote.query.filter_by(
        invoice_id=int(invoice_id), company_id=int(company_id)
    ).all()
    return sum((_money(row.amount_chf) for row in rows), _ZERO).quantize(_CENT)


def split_ht_vat(
    *,
    line: InvoiceLine,
    amount_ttc: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    """Ventile un crédit TTC en HT + TVA proportionnels à la ligne d'origine."""
    ttc = _money(amount_ttc).quantize(_CENT)
    if ttc <= 0:
        raise ValueError("montant_invalide")
    line_ttc = _money(line.total_with_vat)
    line_ht = _money(line.line_total)
    line_vat = _money(line.vat_amount)
    if line_ttc <= 0:
        raise ValueError("ligne_sans_montant")
    if ttc > line_ttc + _CENT:
        raise ValueError("avoir_au_dela_de_la_ligne")
    if ttc == line_ttc:
        return (
            line_ht.quantize(_CENT),
            line_vat.quantize(_CENT),
            line_ttc.quantize(_CENT),
        )
    ratio = ttc / line_ttc
    ht = (line_ht * ratio).quantize(_CENT)
    vat = (ttc - ht).quantize(_CENT)
    if ht < 0 or vat < 0:
        raise ValueError("montant_invalide")
    return ht, vat, ttc


def line_credit_residuals(invoice: Invoice) -> list[dict[str, Any]]:
    """État créditable de chaque ligne (montant calculé serveur)."""
    credited = credited_ttc_by_line(int(invoice.id))
    room = (
        _money(invoice.total_amount)
        - invoice_credit_total(invoice.id, invoice.company_id)
    ).quantize(_CENT)
    if room < 0:
        room = _ZERO
    out: list[dict[str, Any]] = []
    for line in invoice.lines or []:
        line_ttc = _money(line.total_with_vat).quantize(_CENT)
        line_ht = _money(line.line_total).quantize(_CENT)
        line_vat = _money(line.vat_amount).quantize(_CENT)
        already = credited.get(int(line.id), _ZERO).quantize(_CENT)
        residual_ttc = max(_ZERO, (line_ttc - already).quantize(_CENT))
        residual_ttc = min(residual_ttc, room)
        residual_ht = _ZERO
        residual_vat = _ZERO
        if residual_ttc > 0 and line_ttc > 0:
            residual_ht, residual_vat, residual_ttc = split_ht_vat(
                line=line, amount_ttc=residual_ttc
            )
        meta = line.line_meta if isinstance(line.line_meta, dict) else {}
        service_date = meta.get("service_date") or meta.get("service_date_iso")
        out.append(
            {
                "invoice_line_id": int(line.id),
                "description": str(line.description or ""),
                "reservation_id": (
                    int(line.reservation_id)
                    if line.reservation_id is not None
                    else None
                ),
                "service_date": service_date,
                "qty": float(_money(line.qty)),
                "unit_price": float(_money(line.unit_price)),
                "amount_ht": float(line_ht),
                "vat_rate": float(line.vat_rate) if line.vat_rate is not None else None,
                "vat_amount": float(line_vat),
                "amount_ttc": float(line_ttc),
                "credited_ttc": float(already),
                "residual_ht": float(residual_ht),
                "residual_vat": float(residual_vat),
                "residual_ttc": float(residual_ttc),
            }
        )
    return out


def preview_credit_for_lines(
    invoice: Invoice,
    selections: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calcule l'avoir proposé pour une sélection de lignes."""
    residuals = {row["invoice_line_id"]: row for row in line_credit_residuals(invoice)}
    lines_by_id = {int(line.id): line for line in (invoice.lines or [])}
    allocated: list[dict[str, Any]] = []
    total_ttc = _ZERO
    total_ht = _ZERO
    total_vat = _ZERO

    if not selections:
        raise ValueError("lignes_requises")

    seen: set[int] = set()
    for raw in selections:
        try:
            line_id = int(raw.get("invoice_line_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("ligne_inconnue") from exc
        if line_id in seen:
            raise ValueError("ligne_en_double")
        seen.add(line_id)
        line = lines_by_id.get(line_id)
        residual = residuals.get(line_id)
        if line is None or residual is None:
            raise ValueError("ligne_inconnue")
        residual_ttc = _money(residual["residual_ttc"])
        if residual_ttc <= 0:
            raise ValueError("ligne_deja_creditee")
        raw_amount = raw.get("amount_ttc", None)
        if raw_amount is None or raw_amount == "":
            amount_ttc = residual_ttc
        else:
            amount_ttc = _money(raw_amount).quantize(_CENT)
        if amount_ttc <= 0:
            raise ValueError("montant_invalide")
        if amount_ttc > residual_ttc + _CENT:
            raise ValueError("avoir_au_dela_de_la_ligne")
        amount_ttc = min(amount_ttc, residual_ttc)
        ht, vat, ttc = split_ht_vat(line=line, amount_ttc=amount_ttc)
        allocated.append(
            {
                "invoice_line_id": line_id,
                "description": residual["description"],
                "reservation_id": residual["reservation_id"],
                "service_date": residual["service_date"],
                "amount_ht": float(ht),
                "vat_amount": float(vat),
                "amount_ttc": float(ttc),
                "vat_rate": residual["vat_rate"],
                "residual_ttc_before": float(residual_ttc),
            }
        )
        total_ttc += ttc
        total_ht += ht
        total_vat += vat

    total_ttc = total_ttc.quantize(_CENT)
    total_ht = total_ht.quantize(_CENT)
    total_vat = total_vat.quantize(_CENT)
    original = _money(invoice.total_amount)
    already = invoice_credit_total(invoice.id, invoice.company_id)
    room_notes = [total_ttc]
    if already > 0:
        room_notes.insert(0, already)
    try:
        credit_room(invoice.total_amount, room_notes)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    credited_after = (already + total_ttc).quantize(_CENT)
    net = (original - credited_after).quantize(_CENT)
    paid = _money(invoice.amount_paid)
    balance = max(_ZERO, (net - paid).quantize(_CENT))
    overpay = max(_ZERO, (paid - net).quantize(_CENT))
    return {
        "invoice_id": int(invoice.id),
        "invoice_number": invoice.invoice_number,
        "original_total": float(original),
        "credit_notes_total_before": float(already),
        "proposed_amount_ht": float(total_ht),
        "proposed_amount_vat": float(total_vat),
        "proposed_amount_ttc": float(total_ttc),
        "credit_notes_total_after": float(credited_after),
        "net_invoiced_total": float(net),
        "amount_paid": float(paid),
        "balance_due": float(balance),
        "overpayment_amount": float(overpay),
        "lines": allocated,
        "line_residuals": list(residuals.values()),
    }
