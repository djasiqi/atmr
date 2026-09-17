"""Lignes snapshot des factures partenaires (source ≠ ligne facturée)."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from ext import db
from infrastructure.invoices.invoice_calculator import (
    InvoiceCalculator,
    round_to_5_cents,
)
from models.booking_transfer import BookingTransfer
from models.partner_invoice import (
    PartnerInvoice,
    PartnerInvoiceLine,
    partner_invoice_transfers,
)
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)


def _as_decimal(value: Any, default: str = "0") -> Decimal:
    if value is None or value == "":
        return Decimal(default)
    return Decimal(str(value))


def _location_label(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def describe_transfer_line(
    transfer: BookingTransfer, *, amount: Decimal, sort_order: int
) -> dict[str, Any]:
    """Construit un snapshot de ligne à partir d'un transfert source."""
    booking = getattr(transfer, "booking", None)
    date_str = ""
    client_name = "Client"
    departure = ""
    arrival = ""
    if booking is not None:
        if booking.scheduled_time is not None:
            date_str = booking.scheduled_time.strftime("%d.%m.%Y")
        if booking.client and booking.client.user:
            client_name = (
                booking.customer_name
                or f"{booking.client.user.first_name or ''} {booking.client.user.last_name or ''}".strip()
                or booking.client.user.username
                or "Client"
            )
        else:
            client_name = booking.customer_name or "Client"
        departure = _location_label(booking.pickup_location)
        arrival = _location_label(booking.dropoff_location)
    description = f"{client_name} — {departure} → {arrival}".strip(" —→")
    if not description:
        description = client_name
    return {
        "description": description[:500],
        "quantity": Decimal("1"),
        "unit_price": amount,
        "amount": amount,
        "source_type": "booking_transfer",
        "source_id": transfer.id,
        "sort_order": sort_order,
        "service_date": date_str or None,
        "client_name": client_name[:200],
        "departure": departure[:500] if departure else None,
        "arrival": arrival[:500] if arrival else None,
        "note": None,
    }


def load_partner_invoice_transfers(
    partner_invoice: PartnerInvoice,
) -> list[BookingTransfer]:
    """Transferts liés à la facture partenaire (ordre d'insertion)."""
    return (
        db.session.query(BookingTransfer)
        .join(
            partner_invoice_transfers,
            BookingTransfer.id == partner_invoice_transfers.c.booking_transfer_id,
        )
        .filter(partner_invoice_transfers.c.partner_invoice_id == partner_invoice.id)
        .all()
    )


def persist_partner_invoice_lines_from_transfers(
    partner_invoice: PartnerInvoice,
    transfers: list[BookingTransfer],
    *,
    line_amounts: dict[int, Decimal] | None = None,
) -> list[PartnerInvoiceLine]:
    """Crée les lignes snapshot à la génération. N'écrase pas des lignes existantes."""
    if partner_invoice.lines:
        return list(partner_invoice.lines)
    amounts = line_amounts or {}
    created: list[PartnerInvoiceLine] = []
    for index, transfer in enumerate(transfers):
        raw = amounts.get(transfer.id)
        if raw is not None:
            amount = round_to_5_cents(_as_decimal(raw))
        elif transfer.partner_cost:
            amount = round_to_5_cents(_as_decimal(transfer.partner_cost))
        else:
            amount = Decimal("0")
        payload = describe_transfer_line(transfer, amount=amount, sort_order=index)
        line = PartnerInvoiceLine(
            description=payload["description"],
            quantity=payload["quantity"],
            unit_price=payload["unit_price"],
            amount=payload["amount"],
            source_type=payload["source_type"],
            source_id=payload["source_id"],
            sort_order=payload["sort_order"],
            service_date=payload["service_date"],
            client_name=payload["client_name"],
            departure=payload["departure"],
            arrival=payload["arrival"],
        )
        partner_invoice.lines.append(line)
        created.append(line)
    db.session.flush()
    return created


def ensure_partner_invoice_lines(
    partner_invoice: PartnerInvoice,
) -> list[PartnerInvoiceLine]:
    """Persiste un snapshot depuis les transferts si la facture n'a pas encore de lignes."""
    if partner_invoice.lines:
        return list(partner_invoice.lines)
    transfers = load_partner_invoice_transfers(partner_invoice)
    return persist_partner_invoice_lines_from_transfers(partner_invoice, transfers)


def serialize_partner_invoice_lines(
    partner_invoice: PartnerInvoice,
) -> list[dict[str, Any]]:
    """Lignes persistées, ou snapshot mémoire depuis les transferts (GET sans écriture)."""
    if partner_invoice.lines:
        return [line.to_dict() for line in partner_invoice.lines]
    transfers = load_partner_invoice_transfers(partner_invoice)
    rows: list[dict[str, Any]] = []
    for index, transfer in enumerate(transfers):
        amount = (
            round_to_5_cents(_as_decimal(transfer.partner_cost))
            if transfer.partner_cost
            else Decimal("0")
        )
        payload = describe_transfer_line(transfer, amount=amount, sort_order=index)
        payload["id"] = None
        payload["partner_invoice_id"] = partner_invoice.id
        payload["vat_rate"] = None
        payload["quantity"] = float(payload["quantity"])
        payload["unit_price"] = float(payload["unit_price"])
        payload["amount"] = float(payload["amount"])
        rows.append(payload)
    return rows


def apply_partner_invoice_line_updates(
    partner_invoice: PartnerInvoice, lines_payload: list[dict[str, Any]]
) -> None:
    """Met à jour les snapshots. Ne touche pas au transfert source."""
    ensure_partner_invoice_lines(partner_invoice)
    by_id = {line.id: line for line in partner_invoice.lines}
    by_source = {
        int(line.source_id): line
        for line in partner_invoice.lines
        if line.source_id is not None
    }
    ordered = list(partner_invoice.lines)
    for index, raw in enumerate(lines_payload):
        line = None
        line_id = raw.get("id")
        if line_id is not None:
            line = by_id.get(int(line_id))
        if line is None and raw.get("source_id") is not None:
            line = by_source.get(int(raw["source_id"]))
        if line is None and index < len(ordered):
            line = ordered[index]
        if line is None:
            continue
        if "description" in raw and raw["description"] is not None:
            line.description = str(raw["description"])[:500]
            line.client_name = str(raw["description"])[:200]
        if "quantity" in raw:
            line.quantity = round_to_5_cents(_as_decimal(raw["quantity"], "1"))
        if "unit_price" in raw:
            line.unit_price = round_to_5_cents(_as_decimal(raw["unit_price"]))
        if "amount" in raw:
            line.amount = round_to_5_cents(_as_decimal(raw["amount"]))
        elif "quantity" in raw or "unit_price" in raw:
            line.amount = round_to_5_cents(line.quantity * line.unit_price)
        if "note" in raw:
            line.note = str(raw["note"])[:500] if raw["note"] else None
        if "service_date" in raw:
            line.service_date = (
                str(raw["service_date"])[:20] if raw["service_date"] else None
            )
        if "departure" in raw:
            line.departure = str(raw["departure"])[:500] if raw["departure"] else None
        if "arrival" in raw:
            line.arrival = str(raw["arrival"])[:500] if raw["arrival"] else None
        line.sort_order = int(raw.get("sort_order", index))
    db.session.flush()


def recalculate_partner_invoice_totals(partner_invoice: PartnerInvoice) -> None:
    """Recalcule HT / TVA / TTC depuis les lignes snapshot."""
    subtotal = sum((line.amount or Decimal("0")) for line in partner_invoice.lines)
    subtotal = round_to_5_cents(_as_decimal(subtotal))
    billing_settings = CompanyBillingSettingsRepository().find_or_create(
        partner_invoice.executing_company_id
    )
    vat_rate = Decimal(str(getattr(billing_settings, "vat_rate", 0) or 0))
    vat_applicable = (
        bool(getattr(billing_settings, "vat_applicable", False)) and vat_rate > 0
    )
    if not vat_applicable:
        vat_rate = Decimal("0")
    vat_amount, total = InvoiceCalculator().calculate_vat(subtotal, vat_rate)
    partner_invoice.subtotal_amount = subtotal
    partner_invoice.vat_amount = vat_amount
    partner_invoice.total_amount = total


def line_snapshots_for_pdf(partner_invoice: PartnerInvoice) -> list[dict[str, Any]]:
    """Lignes pour le PDF : persistées en priorité."""
    return serialize_partner_invoice_lines(partner_invoice)


def line_amounts_for_pdf(partner_invoice: PartnerInvoice) -> dict[int, Decimal]:
    """Montants par transfer_id pour cohérence PDF historique."""
    amounts: dict[int, Decimal] = {}
    for line in partner_invoice.lines or []:
        if line.source_id is not None:
            amounts[int(line.source_id)] = _as_decimal(line.amount)
    return amounts
