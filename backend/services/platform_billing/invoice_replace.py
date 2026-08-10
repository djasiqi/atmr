"""Remplacement / édition documentaire des factures plateforme (atomique)."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from ext import db
from models.enums import (
    PlatformIssuedDocumentType,
    PlatformIssuedInvoiceStatus,
)
from models.platform_billing import (
    PlatformBillingCreditor,
    PlatformInvoice,
    PlatformIssuedInvoice,
)
from services.platform_billing.decimal_json import decimal_to_str
from services.platform_billing.invoice_pdf import (
    build_platform_qrr_reference,
    build_platform_scor_reference,
    generate_platform_invoice_pdf_bytes,
)
from services.platform_billing.issuance import (
    _build_and_store_pdf,
    _creditor_party,
    _debtor_party,
    _next_invoice_number,
)
from services.platform_billing.money import money_round_chf
from services.platform_billing.payments import (
    _create_credit_note_no_commit,
    _lock_invoice,
)
from services.platform_billing.swiss_qr import (
    SwissQrBillPayload,
    platform_qr_amount,
    render_swiss_qr_bill,
    resolve_platform_reference_mode,
)

CALC_UNIT = "UNIT_PRICE"
CALC_FIXED = "FIXED_AMOUNT"

_INACTIVE = {
    PlatformIssuedInvoiceStatus.CANCELLED.value,
    PlatformIssuedInvoiceStatus.CREDITED.value,
}


class InvoiceReplaceConflict(Exception):
    """Conflit d'optimistic lock (source_updated_at)."""


class InvoiceReplaceError(ValueError):
    """Erreur métier de remplacement."""


def _fmt_qty_label(value: Decimal) -> str:
    s = f"{value.quantize(Decimal('0.01')):.2f}".rstrip("0").rstrip(".")
    return s or "0"


def sync_derived_line_label(
    *,
    label: str,
    line_type: str,
    quantity: Decimal | None,
    unit_amount: Decimal | None,
) -> str:
    """Resynchronise les libellés auto (ex. support X h) avec qté / prix unitaires."""
    lab = (label or "").strip()
    lt = (line_type or "").lower()
    is_support = "support" in lt or lab.lower().startswith("support")
    if is_support and quantity is not None:
        hours_s = _fmt_qty_label(Decimal(str(quantity)))
        if unit_amount is not None:
            rate_s = _fmt_qty_label(Decimal(str(unit_amount)))
            return f"Support plateforme — {hours_s} h à {rate_s} CHF/h"
        return f"Support plateforme — {hours_s} h"
    return lab or "Ligne"


def normalize_editor_lines(
    raw_lines: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Calcule les montants ligne (SSOT serveur)."""
    if not raw_lines:
        raise InvoiceReplaceError("Au moins une ligne est requise")
    normalized: list[dict[str, Any]] = []
    for idx, raw in enumerate(raw_lines):
        if not isinstance(raw, dict):
            raise InvoiceReplaceError(f"Ligne {idx + 1} invalide")
        label = (raw.get("label") or "").strip()
        if not label:
            raise InvoiceReplaceError(f"Ligne {idx + 1} : libellé requis")
        mode = (raw.get("calculation_mode") or CALC_FIXED).strip().upper()
        line_type = (raw.get("line_type") or "ADJUSTMENT").strip() or "ADJUSTMENT"
        try:
            if mode == CALC_UNIT:
                qty = Decimal(str(raw.get("quantity")))
                unit = Decimal(str(raw.get("unit_amount")))
                amount = money_round_chf(qty * unit)
                synced = sync_derived_line_label(
                    label=label,
                    line_type=line_type,
                    quantity=qty,
                    unit_amount=unit,
                )
                normalized.append(
                    {
                        "calculation_mode": CALC_UNIT,
                        "label": synced[:255],
                        "line_type": line_type[:32],
                        "quantity": str(qty),
                        "unit_amount": str(unit),
                        "amount": str(amount),
                    }
                )
            elif mode == CALC_FIXED:
                amount = money_round_chf(Decimal(str(raw.get("amount"))))
                synced = sync_derived_line_label(
                    label=label,
                    line_type=line_type,
                    quantity=None,
                    unit_amount=None,
                )
                normalized.append(
                    {
                        "calculation_mode": CALC_FIXED,
                        "label": synced[:255],
                        "line_type": line_type[:32],
                        "quantity": None,
                        "unit_amount": None,
                        "amount": str(amount),
                    }
                )
            else:
                raise InvoiceReplaceError(
                    f"Ligne {idx + 1} : calculation_mode invalide ({mode})"
                )
        except (InvalidOperation, TypeError, ValueError) as exc:
            raise InvoiceReplaceError(f"Ligne {idx + 1} : montants invalides") from exc
    return normalized


def compute_totals(
    lines: list[dict[str, Any]], tax_rate: Decimal
) -> dict[str, Decimal]:
    subtotal = money_round_chf(
        sum((Decimal(str(ln["amount"])) for ln in lines), Decimal("0.00"))
    )
    tax_amount = money_round_chf(subtotal * tax_rate / Decimal("100"))
    total = money_round_chf(subtotal + tax_amount)
    return {
        "subtotal_amount": subtotal,
        "tax_amount": tax_amount,
        "total_amount": total,
        "qr_amount": platform_qr_amount(total),
    }


def editor_mode_for(inv: PlatformIssuedInvoice) -> str:
    paid = money_round_chf(Decimal(str(inv.amount_paid or 0)))
    if inv.status in _INACTIVE:
        return "readonly"
    if paid > 0:
        return "payments_block"
    if inv.sent_at:
        return "correct"
    return "edit"


def get_editor_bootstrap(issued_id: int) -> dict[str, Any]:
    inv = db.session.get(PlatformIssuedInvoice, int(issued_id))
    if inv is None:
        raise InvoiceReplaceError("Facture introuvable")
    if inv.document_type == PlatformIssuedDocumentType.CREDIT_NOTE.value:
        raise InvoiceReplaceError("Un avoir n'est pas éditable")

    mode = editor_mode_for(inv)
    lines = inv.lines_snapshot if isinstance(inv.lines_snapshot, list) else []
    statement_lines: list[dict[str, Any]] = []
    statement = inv.statement
    if statement is not None:
        for ln in sorted(statement.lines or [], key=lambda x: (x.sort_order, x.id)):
            statement_lines.append(
                {
                    "line_type": ln.line_type,
                    "label": ln.label,
                    "quantity": decimal_to_str(ln.quantity)
                    if ln.quantity is not None
                    else None,
                    "unit_amount": decimal_to_str(ln.unit_amount)
                    if ln.unit_amount is not None
                    else None,
                    "amount": decimal_to_str(ln.amount),
                    "calculation_mode": (
                        CALC_UNIT
                        if ln.quantity is not None and ln.unit_amount is not None
                        else CALC_FIXED
                    ),
                }
            )
        if not lines:
            lines = statement_lines

    return {
        "issued_id": inv.id,
        "invoice_number": inv.invoice_number,
        "mode": mode,
        "sent_at": inv.sent_at.isoformat() if inv.sent_at else None,
        "amount_paid": decimal_to_str(inv.amount_paid),
        "source_updated_at": inv.updated_at.isoformat() if inv.updated_at else None,
        "debtor_snapshot": inv.debtor_snapshot or {},
        "creditor_snapshot": inv.creditor_snapshot or {},
        "lines": lines,
        "statement_lines": statement_lines,
        "tax_rate": decimal_to_str(inv.tax_rate),
        "due_at": inv.due_at.isoformat() if inv.due_at else None,
        "commercial_reference": inv.commercial_reference,
        "billing_year": inv.billing_year,
        "billing_month": inv.billing_month,
        "company_id": inv.company_id,
        "statement_id": inv.statement_id,
        "totals": {
            "subtotal_amount": decimal_to_str(inv.subtotal_amount),
            "tax_amount": decimal_to_str(inv.tax_amount),
            "total_amount": decimal_to_str(inv.total_amount),
        },
    }


def preview_editor_pdf(issued_id: int, payload: dict[str, Any]) -> bytes:
    """PDF aperçu non payable — aucune écriture DB, aucune séquence."""
    inv = db.session.get(PlatformIssuedInvoice, int(issued_id))
    if inv is None:
        raise InvoiceReplaceError("Facture introuvable")
    mode = editor_mode_for(inv)
    if mode in ("readonly", "payments_block"):
        raise InvoiceReplaceError(
            "Correction impossible tant que des paiements sont enregistrés"
            if mode == "payments_block"
            else "Facture en lecture seule"
        )

    lines = normalize_editor_lines(payload.get("lines"))
    tax_rate = Decimal(str(payload.get("tax_rate", inv.tax_rate)))
    totals = compute_totals(lines, tax_rate)
    if totals["total_amount"] <= 0:
        raise InvoiceReplaceError("Le total doit être strictement positif")

    debtor_snap = payload.get("debtor_snapshot") or inv.debtor_snapshot or {}
    creditor_snap = inv.creditor_snapshot or {}
    due_raw = payload.get("due_at")
    due_at = inv.due_at
    if due_raw:
        due_at = datetime.fromisoformat(str(due_raw).replace("Z", "+00:00"))

    pdf_lines = [
        {
            "line_type": ln["line_type"],
            "label": ln["label"],
            "amount": Decimal(str(ln["amount"])),
            "quantity": (
                Decimal(str(ln["quantity"])) if ln.get("quantity") is not None else None
            ),
            "unit_amount": (
                Decimal(str(ln["unit_amount"]))
                if ln.get("unit_amount") is not None
                else None
            ),
        }
        for ln in lines
    ]
    return generate_platform_invoice_pdf_bytes(
        invoice_number="APERÇU",
        issued_at=inv.issued_at or datetime.now(UTC),
        due_at=due_at,
        period_year=int(inv.billing_year or 0),
        period_month=int(inv.billing_month or 0),
        creditor_snap=creditor_snap,
        debtor_snap=debtor_snap,
        lines=pdf_lines,
        subtotal=totals["subtotal_amount"],
        tax_rate=tax_rate,
        tax_amount=totals["tax_amount"],
        total=totals["total_amount"],
        qr_amount=totals["qr_amount"],
        qr_reference=None,
        payment_reference_mode="NON",
        iban="",
        payment_terms_days=30,
        preview=True,
    )


def _parse_source_updated_at(raw: str | None) -> datetime | None:
    if not raw:
        return None
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))


def _cancel_no_commit(inv: PlatformIssuedInvoice) -> None:
    if inv.status not in (
        PlatformIssuedInvoiceStatus.DRAFT.value,
        PlatformIssuedInvoiceStatus.ISSUED.value,
    ):
        raise InvoiceReplaceError("Annulation impossible après envoi/paiement")
    if inv.sent_at:
        raise InvoiceReplaceError("Annulation impossible après envoi")
    if money_round_chf(Decimal(str(inv.amount_paid or 0))) > 0:
        raise InvoiceReplaceError(
            "Annulation impossible : des paiements sont enregistrés"
        )
    inv.status = PlatformIssuedInvoiceStatus.CANCELLED.value
    inv.cancelled_at = datetime.now(UTC)


def _issue_from_snapshot_no_commit(
    *,
    source: PlatformIssuedInvoice,
    lines: list[dict[str, Any]],
    totals: dict[str, Decimal],
    tax_rate: Decimal,
    debtor_snap: dict[str, Any],
    due_at: datetime | None,
    commercial_reference: str | None,
    replaces_id: int,
    replace_idempotency_key: str,
) -> PlatformIssuedInvoice:
    creditor = PlatformBillingCreditor.query.filter_by(is_active=True).first()
    if not creditor:
        raise InvoiceReplaceError("Créancier LIRIE manquant")
    _, creditor_snap, iban = _creditor_party(creditor)

    year = int(source.billing_year or 0)
    month = int(source.billing_month or 0)
    if not year or not month:
        raise InvoiceReplaceError("Période de facturation manquante")
    invoice_number = _next_invoice_number(year, month)
    now = datetime.now(UTC)

    inv = PlatformIssuedInvoice(
        statement_id=source.statement_id,
        company_id=source.company_id,
        invoice_number=invoice_number,
        document_type=PlatformIssuedDocumentType.INVOICE.value,
        status=PlatformIssuedInvoiceStatus.ISSUED.value,
        currency=source.currency or "CHF",
        subtotal_amount=totals["subtotal_amount"],
        tax_rate=tax_rate,
        tax_amount=totals["tax_amount"],
        total_amount=totals["total_amount"],
        qr_amount=totals["qr_amount"],
        issued_at=now,
        due_at=due_at or source.due_at,
        debtor_snapshot=debtor_snap,
        creditor_snapshot=creditor_snap or source.creditor_snapshot,
        lines_snapshot=lines,
        billing_config_id=source.billing_config_id,
        partner_agreement_id=source.partner_agreement_id,
        dunning_policy_snapshot=source.dunning_policy_snapshot,
        dunning_automation_authorized_at_issuance=bool(
            source.dunning_automation_authorized_at_issuance
        ),
        billing_year=year,
        billing_month=month,
        period_id=source.period_id,
        replaces_issued_invoice_id=replaces_id,
        replace_idempotency_key=replace_idempotency_key,
        commercial_reference=(commercial_reference or None),
        amount_paid=Decimal("0.00"),
    )
    db.session.add(inv)
    db.session.flush()

    from models import Company
    from models.billing_profile import CompanyBillingProfile

    debtor_party = None
    if debtor_snap.get("street_name"):
        from services.platform_billing.swiss_qr import QrParty

        debtor_party = QrParty(
            name=str(debtor_snap.get("legal_name") or ""),
            street=str(debtor_snap.get("street_name") or ""),
            building_number=debtor_snap.get("building_number"),
            postal_code=str(debtor_snap.get("postal_code") or ""),
            city=str(debtor_snap.get("city") or ""),
            country_code=str(debtor_snap.get("country_code") or "CH"),
        )
    else:
        company = db.session.get(Company, source.company_id)
        profile = CompanyBillingProfile.query.filter_by(
            company_id=source.company_id
        ).first()
        if company:
            debtor_party, debtor_snap = _debtor_party(company, profile)
            inv.debtor_snapshot = debtor_snap

    creditor_party, _, _ = _creditor_party(creditor)
    ref_mode = resolve_platform_reference_mode(
        iban, creditor.payment_reference_mode or "QRR"
    )
    if ref_mode == "QRR":
        qr_reference = build_platform_qrr_reference(
            invoice_number=invoice_number,
            issued_id=inv.id,
            creditor_reference_base=getattr(creditor, "creditor_reference_base", None)
            or "21",
        )
    elif ref_mode == "SCOR":
        qr_reference = build_platform_scor_reference(invoice_number=invoice_number)
    else:
        qr_reference = None
    inv.qr_reference = qr_reference

    if debtor_party is not None:
        payload = SwissQrBillPayload(
            creditor=creditor_party,
            debtor=debtor_party,
            iban=iban,
            reference_type=ref_mode,
            reference=qr_reference,
            amount=totals["qr_amount"],
            currency="CHF",
            additional_information=invoice_number,
        )
        try:
            render_swiss_qr_bill(payload)
        except Exception as e:
            raise InvoiceReplaceError(f"Émission QR impossible: {e}") from e

    statement = (
        db.session.get(PlatformInvoice, inv.statement_id) if inv.statement_id else None
    )
    days = creditor.payment_terms_days_default or 30
    _build_and_store_pdf(
        inv=inv,
        statement=statement,
        creditor=creditor,
        debtor_snap=inv.debtor_snapshot or {},
        creditor_snap=inv.creditor_snapshot or {},
        iban=iban,
        payment_terms_days=int(days),
    )
    return inv


def replace_issued_invoice(
    issued_id: int,
    payload: dict[str, Any],
    *,
    admin_user_id: int | None = None,
) -> PlatformIssuedInvoice:
    """Remplace une facture (édition avant envoi ou correction après envoi unpaid)."""
    idem = (payload.get("idempotency_key") or "").strip()
    if not idem:
        raise InvoiceReplaceError("idempotency_key est requis")

    existing_by_key = PlatformIssuedInvoice.query.filter_by(
        replace_idempotency_key=idem
    ).first()
    if existing_by_key is not None:
        return existing_by_key

    source = _lock_invoice(int(issued_id))
    if source.document_type != PlatformIssuedDocumentType.INVOICE.value:
        raise InvoiceReplaceError("Seule une facture peut être remplacée")

    mode = editor_mode_for(source)
    if mode == "readonly":
        raise InvoiceReplaceError("Facture en lecture seule")
    if mode == "payments_block":
        raise InvoiceReplaceError(
            "Correction impossible tant que des paiements sont enregistrés"
        )

    expected = _parse_source_updated_at(payload.get("source_updated_at"))
    if expected is None:
        raise InvoiceReplaceError("source_updated_at est requis")
    actual = source.updated_at
    if actual is not None:
        act = actual if actual.tzinfo else actual.replace(tzinfo=UTC)
        exp = expected if expected.tzinfo else expected.replace(tzinfo=UTC)
        # Tolérance ms (sérialisation ISO)
        if abs((act - exp).total_seconds()) > 0.001:
            raise InvoiceReplaceConflict(
                "La facture a été modifiée entre-temps — rechargez l'éditeur"
            )

    already = PlatformIssuedInvoice.query.filter_by(
        replaces_issued_invoice_id=source.id
    ).first()
    if already is not None:
        raise InvoiceReplaceError("Cette facture a déjà été remplacée")

    lines = normalize_editor_lines(payload.get("lines"))
    tax_rate = Decimal(str(payload.get("tax_rate", source.tax_rate)))
    totals = compute_totals(lines, tax_rate)
    if totals["total_amount"] <= 0:
        raise InvoiceReplaceError("Le total doit être strictement positif")

    debtor_snap = payload.get("debtor_snapshot") or source.debtor_snapshot or {}
    due_raw = payload.get("due_at")
    due_at = source.due_at
    if due_raw:
        due_at = datetime.fromisoformat(str(due_raw).replace("Z", "+00:00"))
    commercial_reference = payload.get("commercial_reference")
    if commercial_reference is not None:
        commercial_reference = str(commercial_reference).strip()[:128] or None

    reason = (payload.get("reason") or "").strip()
    if source.sent_at and len(reason) < 3:
        raise InvoiceReplaceError("Motif obligatoire (3 car. min.) après envoi")

    if not source.sent_at:
        _cancel_no_commit(source)
    else:
        _create_credit_note_no_commit(
            source,
            reason=reason or "Correction documentaire",
            created_by_user_id=admin_user_id,
        )

    replacement = _issue_from_snapshot_no_commit(
        source=source,
        lines=lines,
        totals=totals,
        tax_rate=tax_rate,
        debtor_snap=debtor_snap,
        due_at=due_at,
        commercial_reference=commercial_reference,
        replaces_id=source.id,
        replace_idempotency_key=idem,
    )
    db.session.commit()
    db.session.refresh(replacement)
    return replacement
