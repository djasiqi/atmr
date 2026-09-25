"""Frais de facture papier (Direct patient / PORTAL) — ligne Invoice distincte."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

PAPER_INVOICE_FEE_CHF = Decimal("3.00")
PAPER_FEE_LINE_META_KEY = "paper_invoice_fee"
PAPER_FEE_DESCRIPTION = "Frais de facture papier"

DELIVERY_EMAIL = "email"
DELIVERY_PAPER = "paper"


def normalize_delivery_method(raw: object | None) -> str:
    value = str(raw or DELIVERY_EMAIL).strip().lower()
    if value in (DELIVERY_PAPER, "courrier", "print"):
        return DELIVERY_PAPER
    return DELIVERY_EMAIL


def invoice_has_paper_fee_line(invoice: Any) -> bool:
    lines = getattr(invoice, "lines", None) or []
    for line in lines:
        meta = getattr(line, "line_meta", None) or {}
        if isinstance(meta, dict) and meta.get(PAPER_FEE_LINE_META_KEY):
            return True
        desc = (getattr(line, "description", None) or "").strip().lower()
        if desc == PAPER_FEE_DESCRIPTION.lower():
            return True
    return False


def resolve_invoice_delivery_method(
    *,
    invoice: Any = None,
    client: Any = None,
    explicit: object | None = None,
) -> str:
    """Résout email|paper : préférence client uniquement (défaut email).

    ``explicit`` est ignoré : l'entreprise de transport ne choisit pas le mode.
    ``invoice.meta.delivery_method`` reste utilisé pour les factures déjà créées.
    """
    del explicit  # non utilisé — contrat produit : préférence client seule
    if client is not None:
        return normalize_delivery_method(
            getattr(client, "invoice_delivery_method", None)
        )
    meta = getattr(invoice, "meta", None) if invoice is not None else None
    if isinstance(meta, dict) and meta.get("delivery_method"):
        return normalize_delivery_method(meta.get("delivery_method"))
    return DELIVERY_EMAIL


def ensure_paper_invoice_fee_line(
    invoice: Any,
    *,
    client: Any = None,
    explicit: object | None = None,
) -> bool:
    """Ajoute la ligne CHF 3 si mode papier et absente. Retourne True si ajoutée.

    Recalcule sous-total / total / solde. À appeler dans une transaction ouverte.
    """
    if invoice is None:
        return False
    delivery = resolve_invoice_delivery_method(
        invoice=invoice, client=client, explicit=explicit
    )
    if delivery != DELIVERY_PAPER:
        return False
    if invoice_has_paper_fee_line(invoice):
        return False

    from ext import db
    from infrastructure.invoices.invoice_calculator import round_to_5_cents
    from models.enums import InvoiceLineType
    from models.invoice import InvoiceLine

    fee = PAPER_INVOICE_FEE_CHF
    line = InvoiceLine()
    line.invoice_id = int(invoice.id)
    line.type = InvoiceLineType.CUSTOM
    line.description = PAPER_FEE_DESCRIPTION
    line.qty = Decimal("1")
    line.unit_price = fee
    line.line_total = fee
    line.vat_rate = None
    line.vat_amount = Decimal("0.00")
    line.total_with_vat = fee
    line.adjustment_note = None
    line.reservation_id = None
    line.line_meta = {
        PAPER_FEE_LINE_META_KEY: True,
        "amount_chf": float(fee),
    }
    db.session.add(line)
    db.session.flush()

    sub = round_to_5_cents((invoice.subtotal_amount or Decimal("0")) + fee)
    total = round_to_5_cents((invoice.total_amount or Decimal("0")) + fee)
    invoice.subtotal_amount = sub
    invoice.total_amount = total
    invoice.balance_due = round_to_5_cents(
        total - (invoice.amount_paid or Decimal("0"))
    )

    current_meta: dict[str, Any] = {}
    if isinstance(invoice.meta, dict):
        current_meta = dict(invoice.meta)
    current_meta["delivery_method"] = DELIVERY_PAPER
    current_meta["paper_invoice_fee_chf"] = float(fee)
    invoice.meta = current_meta

    try:
        db.session.expire(invoice, ["lines"])
    except Exception:
        pass
    return True
