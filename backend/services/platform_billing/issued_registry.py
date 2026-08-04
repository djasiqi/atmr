"""Registre admin des factures légales plateforme (liste, stats, export)."""

from __future__ import annotations

import csv
import io
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.orm import joinedload

from ext import db
from models.company import Company
from models.enums import PlatformIssuedDocumentType, PlatformIssuedInvoiceStatus
from models.platform_billing import (
    PlatformInvoiceDueDateChange,
    PlatformInvoicePayment,
    PlatformIssuedInvoice,
)
from services.platform_billing.decimal_json import decimal_to_str
from services.platform_billing.issued_status import (
    balance_due_for_registry,
    is_overdue_read,
    serialize_issued_status_fields,
)
from services.platform_billing.money import money_round_chf


def serialize_payment(p: PlatformInvoicePayment) -> dict[str, Any]:
    return {
        "id": p.id,
        "entry_type": getattr(p, "entry_type", None) or "PAYMENT",
        "amount": decimal_to_str(p.amount),
        "paid_at": p.paid_at.isoformat() if p.paid_at else None,
        "method": p.method,
        "reference": p.reference,
        "notes": p.notes,
        "idempotency_key": getattr(p, "idempotency_key", None),
        "reverses_payment_id": getattr(p, "reverses_payment_id", None),
        "reversal_reason": getattr(p, "reversal_reason", None),
        "created_by_user_id": p.created_by_user_id,
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


def serialize_due_change(c: PlatformInvoiceDueDateChange) -> dict[str, Any]:
    return {
        "id": c.id,
        "old_due_at": c.old_due_at.isoformat() if c.old_due_at else None,
        "new_due_at": c.new_due_at.isoformat() if c.new_due_at else None,
        "reason": c.reason,
        "change_type": c.change_type,
        "admin_user_id": c.admin_user_id,
        "old_pdf_checksum": c.old_pdf_checksum,
        "new_pdf_checksum": c.new_pdf_checksum,
        "created_at": c.created_at.isoformat() if c.created_at else None,
    }


def _fmt_due(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%d.%m.%Y")


def build_issued_invoice_timeline(inv: PlatformIssuedInvoice) -> list[dict[str, Any]]:
    """Chronologie documentaire reconstruite depuis les faits persistés."""
    events: list[dict[str, Any]] = []

    def add(
        *,
        type_: str,
        at: datetime | None,
        label: str,
        detail: str | None = None,
        seq: int = 50,
        **extra: Any,
    ) -> None:
        if at is None:
            return
        ts = at if at.tzinfo else at.replace(tzinfo=UTC)
        events.append(
            {
                "type": type_,
                "at": ts.isoformat(),
                "label": label,
                "detail": detail,
                "_sort": (ts, seq, type_),
                **extra,
            }
        )

    is_credit = inv.document_type == PlatformIssuedDocumentType.CREDIT_NOTE.value
    issued_label = "Avoir émis" if is_credit else "Facture émise"
    add(
        type_="ISSUED",
        at=inv.issued_at or inv.created_at,
        label=issued_label,
        detail=f"{inv.invoice_number} · {decimal_to_str(inv.total_amount)} {inv.currency}",
        seq=0,
    )

    replaces_id = getattr(inv, "replaces_issued_invoice_id", None)
    if replaces_id:
        prev = db.session.get(PlatformIssuedInvoice, int(replaces_id))
        prev_num = prev.invoice_number if prev else f"#{replaces_id}"
        add(
            type_="REPLACES",
            at=inv.issued_at or inv.created_at,
            label="Remplace une facture",
            detail=prev_num,
            related_invoice_id=int(replaces_id),
            related_invoice_number=prev_num,
            seq=1,
        )

    credit_of = getattr(inv, "credit_of_invoice_id", None)
    if credit_of:
        src = db.session.get(PlatformIssuedInvoice, int(credit_of))
        src_num = src.invoice_number if src else f"#{credit_of}"
        add(
            type_="CREDIT_OF",
            at=inv.issued_at or inv.created_at,
            label="Avoir sur facture",
            detail=src_num,
            related_invoice_id=int(credit_of),
            related_invoice_number=src_num,
            seq=1,
        )

    if inv.due_at and not (getattr(inv, "due_date_changes", None) or []):
        add(
            type_="DUE_SET",
            at=inv.issued_at or inv.created_at,
            label="Échéance",
            detail=_fmt_due(inv.due_at),
            seq=2,
        )

    add(
        type_="SENT",
        at=inv.sent_at,
        label="Marquée comme envoyée",
        detail=None,
        seq=10,
    )

    for c in getattr(inv, "due_date_changes", None) or []:
        old_s = _fmt_due(c.old_due_at) or "—"
        new_s = _fmt_due(c.new_due_at) or "—"
        detail = f"{old_s} → {new_s}"
        if c.reason:
            detail = f"{detail} · {c.reason}"
        add(
            type_="DUE_CHANGED",
            at=c.created_at,
            label="Échéance modifiée",
            detail=detail,
            seq=20,
        )

    for p in inv.payments or []:
        entry = getattr(p, "entry_type", None) or "PAYMENT"
        if entry == "REVERSAL":
            bits = [f"{decimal_to_str(p.amount)} {inv.currency}"]
            if p.reversal_reason:
                bits.append(p.reversal_reason)
            add(
                type_="PAYMENT_REVERSAL",
                at=p.paid_at or p.created_at,
                label="Paiement contrepassé",
                detail=" · ".join(bits),
                payment_id=p.id,
                seq=30,
            )
        else:
            bits = [f"{decimal_to_str(p.amount)} {inv.currency}"]
            if p.method:
                bits.append(str(p.method))
            if p.reference:
                bits.append(f"réf. {p.reference}")
            add(
                type_="PAYMENT",
                at=p.paid_at or p.created_at,
                label="Paiement enregistré",
                detail=" · ".join(bits),
                payment_id=p.id,
                seq=30,
            )

    credit_note = getattr(inv, "credit_note", None)
    if credit_note is not None:
        bits = [credit_note.invoice_number]
        reason = getattr(inv, "credit_reason", None) or getattr(
            credit_note, "credit_reason", None
        )
        if reason:
            bits.append(reason)
        add(
            type_="CREDITED",
            at=inv.credited_at or credit_note.issued_at or credit_note.created_at,
            label="Avoir créé",
            detail=" · ".join(bits),
            related_invoice_id=credit_note.id,
            related_invoice_number=credit_note.invoice_number,
            seq=40,
        )
    elif inv.credited_at:
        detail = getattr(inv, "credit_reason", None)
        add(
            type_="CREDITED",
            at=inv.credited_at,
            label="Facture créditée",
            detail=detail,
            seq=40,
        )

    replaced_by = getattr(inv, "replaced_by", None)
    if inv.cancelled_at:
        label = "Facture annulée"
        detail = None
        if replaced_by is not None:
            label = "Annulée (remplacement)"
            detail = f"→ {replaced_by.invoice_number}"
        add(
            type_="CANCELLED",
            at=inv.cancelled_at,
            label=label,
            detail=detail,
            seq=45,
        )

    if replaced_by is not None:
        add(
            type_="REPLACED_BY",
            at=replaced_by.issued_at or replaced_by.created_at or inv.cancelled_at,
            label="Remplacée par",
            detail=replaced_by.invoice_number,
            related_invoice_id=replaced_by.id,
            related_invoice_number=replaced_by.invoice_number,
            seq=46,
        )
    events.sort(key=lambda e: e["_sort"])
    for e in events:
        e.pop("_sort", None)
    return events



def serialize_issued_invoice(
    inv: PlatformIssuedInvoice,
    *,
    company_name: str | None = None,
    include_payments: bool = False,
    include_due_changes: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    now = now or datetime.now(UTC)
    name = company_name
    if name is None and inv.company_id:
        co = db.session.get(Company, inv.company_id)
        name = co.name if co else None
    status_fields = serialize_issued_status_fields(inv, now=now)
    credit_of = getattr(inv, "credit_of_invoice_id", None)
    credit_note = getattr(inv, "credit_note", None)
    data: dict[str, Any] = {
        "id": inv.id,
        "statement_id": inv.statement_id,
        "company_id": inv.company_id,
        "company_name": name,
        "invoice_number": inv.invoice_number,
        "status": inv.status,
        "currency": inv.currency,
        "subtotal_amount": decimal_to_str(inv.subtotal_amount),
        "tax_rate": decimal_to_str(inv.tax_rate, places=4),
        "tax_amount": decimal_to_str(inv.tax_amount),
        "total_amount": decimal_to_str(inv.total_amount),
        "qr_amount": decimal_to_str(inv.qr_amount),
        "qr_reference": inv.qr_reference,
        "issued_at": inv.issued_at.isoformat() if inv.issued_at else None,
        "due_at": inv.due_at.isoformat() if inv.due_at else None,
        "sent_at": inv.sent_at.isoformat() if inv.sent_at else None,
        "paid_at": inv.paid_at.isoformat() if inv.paid_at else None,
        "cancelled_at": inv.cancelled_at.isoformat() if inv.cancelled_at else None,
        "credited_at": inv.credited_at.isoformat() if inv.credited_at else None,
        "credit_of_invoice_id": credit_of,
        "credit_note_id": credit_note.id if credit_note else None,
        "replaces_issued_invoice_id": getattr(
            inv, "replaces_issued_invoice_id", None
        ),
        "billing_year": getattr(inv, "billing_year", None),
        "billing_month": getattr(inv, "billing_month", None),
        "period_id": getattr(inv, "period_id", None),
        "credit_reason": getattr(inv, "credit_reason", None),
        "pdf_storage_key": inv.pdf_storage_key,
        "pdf_checksum": inv.pdf_checksum,
        "amount_paid": decimal_to_str(inv.amount_paid),
        **status_fields,
    }
    if include_payments:
        payments = sorted(
            inv.payments or [],
            key=lambda p: (p.paid_at or p.created_at, p.id),
        )
        data["payments"] = [serialize_payment(p) for p in payments]
    if include_due_changes:
        changes = sorted(
            getattr(inv, "due_date_changes", None) or [],
            key=lambda c: (c.created_at, c.id),
        )
        data["due_date_changes"] = [serialize_due_change(c) for c in changes]
    return data


def _base_query():
    return (
        select(PlatformIssuedInvoice, Company.name)
        .outerjoin(Company, Company.id == PlatformIssuedInvoice.company_id)
    )


def _apply_filters(
    stmt,
    *,
    q: str | None = None,
    company_id: int | None = None,
    status: str | None = None,
    payment_state: str | None = None,
    year: int | None = None,
    month: int | None = None,
    with_balance: bool = False,
    overdue_only: bool = False,
    with_dunning: bool = False,
    document_type: str | None = None,
    now: datetime | None = None,
):
    now = now or datetime.now(UTC)
    if q:
        like = f"%{q.strip()}%"
        stmt = stmt.where(
            or_(
                PlatformIssuedInvoice.invoice_number.ilike(like),
                Company.name.ilike(like),
            )
        )
    if company_id is not None:
        stmt = stmt.where(PlatformIssuedInvoice.company_id == int(company_id))
    if status:
        stmt = stmt.where(PlatformIssuedInvoice.status == status)
    if document_type:
        stmt = stmt.where(PlatformIssuedInvoice.document_type == document_type)
    if year is not None:
        stmt = stmt.where(PlatformIssuedInvoice.billing_year == int(year))
    if month is not None:
        stmt = stmt.where(PlatformIssuedInvoice.billing_month == int(month))
    if with_balance:
        stmt = stmt.where(
            PlatformIssuedInvoice.document_type
            == PlatformIssuedDocumentType.INVOICE.value,
            PlatformIssuedInvoice.amount_paid < PlatformIssuedInvoice.total_amount,
            PlatformIssuedInvoice.status.notin_(
                [
                    PlatformIssuedInvoiceStatus.CANCELLED.value,
                    PlatformIssuedInvoiceStatus.CREDITED.value,
                    PlatformIssuedInvoiceStatus.PAID.value,
                ]
            ),
        )
    if overdue_only:
        stmt = stmt.where(
            PlatformIssuedInvoice.document_type
            == PlatformIssuedDocumentType.INVOICE.value,
            PlatformIssuedInvoice.sent_at.isnot(None),
            PlatformIssuedInvoice.due_at.isnot(None),
            PlatformIssuedInvoice.due_at < now,
            PlatformIssuedInvoice.amount_paid < PlatformIssuedInvoice.total_amount,
            PlatformIssuedInvoice.status.notin_(
                [
                    PlatformIssuedInvoiceStatus.CANCELLED.value,
                    PlatformIssuedInvoiceStatus.CREDITED.value,
                    PlatformIssuedInvoiceStatus.PAID.value,
                ]
            ),
        )
    if with_dunning:
        from models.platform_billing import PlatformDunningCase

        stmt = stmt.where(
            PlatformIssuedInvoice.company_id.in_(
                select(PlatformDunningCase.company_id).where(
                    PlatformDunningCase.status.in_(("open", "partial", "full"))
                )
            )
        )
    # payment_state filtré en Python après sérialisation si PARTIAL
    return stmt, payment_state


def _compute_stats(rows: list[PlatformIssuedInvoice], *, now: datetime) -> dict[str, Any]:
    total_invoiced = Decimal("0.00")
    total_credits = Decimal("0.00")
    total_paid = Decimal("0.00")
    total_balance = Decimal("0.00")
    overdue_count = 0
    overdue_amount = Decimal("0.00")
    for inv in rows:
        dtype = getattr(inv, "document_type", None) or "INVOICE"
        if dtype == PlatformIssuedDocumentType.CREDIT_NOTE.value:
            total_credits += abs(Decimal(str(inv.total_amount or 0)))
            continue
        if inv.status in (
            PlatformIssuedInvoiceStatus.CANCELLED.value,
        ):
            continue
        total = Decimal(str(inv.total_amount or 0))
        paid = Decimal(str(inv.amount_paid or 0))
        if total > 0:
            total_invoiced += total
            total_paid += paid
            bal = balance_due_for_registry(inv)
            if bal > 0:
                total_balance += bal
            if is_overdue_read(inv, now=now):
                overdue_count += 1
                overdue_amount += bal
    return {
        "total_invoiced": decimal_to_str(money_round_chf(total_invoiced)),
        "total_credits": decimal_to_str(money_round_chf(total_credits)),
        "net_invoiced": decimal_to_str(
            money_round_chf(total_invoiced - total_credits)
        ),
        "total_paid": decimal_to_str(money_round_chf(total_paid)),
        "total_balance": decimal_to_str(money_round_chf(total_balance)),
        "overdue_count": overdue_count,
        "overdue_amount": decimal_to_str(money_round_chf(overdue_amount)),
    }


def list_issued_invoices(
    *,
    q: str | None = None,
    company_id: int | None = None,
    status: str | None = None,
    payment_state: str | None = None,
    year: int | None = None,
    month: int | None = None,
    with_balance: bool = False,
    overdue_only: bool = False,
    with_dunning: bool = False,
    document_type: str | None = None,
    page: int = 1,
    per_page: int = 20,
    sort_by: str = "issued_at",
    sort_order: str = "desc",
) -> dict[str, Any]:
    now = datetime.now(UTC)
    page = max(1, int(page or 1))
    per_page = min(100, max(1, int(per_page or 20)))

    stmt, pay_filter = _apply_filters(
        _base_query(),
        q=q,
        company_id=company_id,
        status=status,
        payment_state=payment_state,
        year=year,
        month=month,
        with_balance=with_balance,
        overdue_only=overdue_only,
        with_dunning=with_dunning,
        document_type=document_type,
        now=now,
    )

    sort_col = {
        "issued_at": PlatformIssuedInvoice.issued_at,
        "due_at": PlatformIssuedInvoice.due_at,
        "invoice_number": PlatformIssuedInvoice.invoice_number,
        "total_amount": PlatformIssuedInvoice.total_amount,
        "company_name": Company.name,
    }.get(sort_by, PlatformIssuedInvoice.issued_at)
    if (sort_order or "desc").lower() == "asc":
        stmt = stmt.order_by(sort_col.asc().nullslast(), PlatformIssuedInvoice.id.desc())
    else:
        stmt = stmt.order_by(sort_col.desc().nullslast(), PlatformIssuedInvoice.id.desc())

    all_rows = db.session.execute(stmt).all()
    # Stats sur le jeu filtré (avant pagination / payment_state)
    all_invoices = [r[0] for r in all_rows]
    if pay_filter:
        filtered: list[tuple] = []
        for inv, cname in all_rows:
            from services.platform_billing.issued_status import payment_state as ps

            if ps(inv) == pay_filter.upper():
                filtered.append((inv, cname))
        all_rows = filtered
        all_invoices = [r[0] for r in all_rows]

    stats = _compute_stats(all_invoices, now=now)
    total = len(all_rows)
    pages = max(1, (total + per_page - 1) // per_page) if total else 0
    start = (page - 1) * per_page
    page_rows = all_rows[start : start + per_page]
    items = [
        serialize_issued_invoice(inv, company_name=cname, now=now)
        for inv, cname in page_rows
    ]
    return {
        "items": items,
        "pagination": {
            "page": page,
            "per_page": per_page,
            "total": total,
            "pages": pages,
        },
        "stats": stats,
    }


def get_issued_invoice_detail(issued_id: int) -> dict[str, Any]:
    inv = (
        db.session.query(PlatformIssuedInvoice)
        .options(
            joinedload(PlatformIssuedInvoice.payments),
            joinedload(PlatformIssuedInvoice.due_date_changes),
            joinedload(PlatformIssuedInvoice.statement),
        )
        .filter_by(id=int(issued_id))
        .first()
    )
    if inv is None:
        raise ValueError("Facture introuvable")
    data = serialize_issued_invoice(
        inv, include_payments=True, include_due_changes=True
    )
    # Lignes documentaires (snapshot) + lignes relevé (lecture seule / reset éditeur)
    snap = inv.lines_snapshot
    if isinstance(snap, list) and snap:
        data["lines"] = snap
    else:
        data["lines"] = []

    statement = inv.statement
    if statement is not None:
        lines = sorted(statement.lines or [], key=lambda x: (x.sort_order, x.id))
        data["statement_lines"] = [
            {
                "id": ln.id,
                "line_type": ln.line_type,
                "label": ln.label,
                "quantity": decimal_to_str(ln.quantity) if ln.quantity is not None else None,
                "unit_amount": (
                    decimal_to_str(ln.unit_amount) if ln.unit_amount is not None else None
                ),
                "amount": decimal_to_str(ln.amount),
                "calculation_mode": (
                    "UNIT_PRICE"
                    if ln.quantity is not None and ln.unit_amount is not None
                    else "FIXED_AMOUNT"
                ),
            }
            for ln in lines
        ]
        if not data["lines"]:
            data["lines"] = [
                {
                    "line_type": ln["line_type"],
                    "label": ln["label"],
                    "quantity": ln["quantity"],
                    "unit_amount": ln["unit_amount"],
                    "amount": ln["amount"],
                    "calculation_mode": ln["calculation_mode"],
                }
                for ln in data["statement_lines"]
            ]
    else:
        data["statement_lines"] = []

    data["commercial_reference"] = getattr(inv, "commercial_reference", None)
    data["source_updated_at"] = (
        inv.updated_at.isoformat() if inv.updated_at else None
    )
    data["timeline"] = build_issued_invoice_timeline(inv)

    # Dunning résumé
    from models.platform_billing import PlatformDunningCase, PlatformInvoiceDunningHold

    holds = (
        PlatformInvoiceDunningHold.query.filter_by(
            issued_invoice_id=inv.id, released_at=None
        ).all()
    )
    data["dunning_holds"] = [
        {
            "id": h.id,
            "reason": h.reason,
            "disputed_amount": decimal_to_str(h.disputed_amount),
            "hold_until": h.hold_until.isoformat() if h.hold_until else None,
        }
        for h in holds
    ]
    case = (
        PlatformDunningCase.query.filter_by(company_id=inv.company_id)
        .filter(PlatformDunningCase.status.in_(("open", "partial", "full")))
        .order_by(PlatformDunningCase.id.desc())
        .first()
    )
    data["dunning_case"] = (
        {
            "id": case.id,
            "status": case.status,
            "opened_at": case.opened_at.isoformat() if case.opened_at else None,
        }
        if case
        else None
    )
    return data


def export_issued_invoices_csv(**filters) -> tuple[str, str]:
    """Retourne (csv_text, filename) avec les mêmes filtres que la liste."""
    result = list_issued_invoices(**filters, page=1, per_page=10000)
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";")
    writer.writerow(
        [
            "numero",
            "document_type",
            "entreprise",
            "periode",
            "statut",
            "ui_status",
            "emission",
            "echeance",
            "total_ttc",
            "paye",
            "solde",
            "payment_state",
        ]
    )
    for it in result["items"]:
        period = ""
        if it.get("billing_year") and it.get("billing_month"):
            period = f"{it['billing_year']}-{int(it['billing_month']):02d}"
        writer.writerow(
            [
                it.get("invoice_number"),
                it.get("document_type"),
                it.get("company_name"),
                period,
                it.get("status"),
                it.get("ui_status"),
                (it.get("issued_at") or "")[:10],
                (it.get("due_at") or "")[:10],
                it.get("total_amount"),
                it.get("amount_paid"),
                it.get("balance_due"),
                it.get("payment_state"),
            ]
        )
    return buf.getvalue(), "platform-issued-invoices.csv"
