"""Lignes snapshot des factures partenaires (source ≠ ligne facturée)."""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Any

from application.invoices.partner_pickup_time import (
    SOURCE_UNKNOWN,
    pickup_display_label,
    snapshot_pickup_from_booking,
)
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


class PartnerDraftEditError(Exception):
    """Refus métier d'une commande d'édition partenaire."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


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
    pickup = snapshot_pickup_from_booking(booking)
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
        "scheduled_pickup_at": pickup["scheduled_pickup_at"],
        "boarded_at": pickup["boarded_at"],
        "pickup_time_source": pickup["pickup_time_source"],
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
            scheduled_pickup_at=payload["scheduled_pickup_at"],
            boarded_at=payload["boarded_at"],
            pickup_time_source=payload["pickup_time_source"],
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


def note_state(line: PartnerInvoiceLine) -> dict[str, Any]:
    """Note utilisateur, ou état JSON versionné (remise / prestation)."""
    raw = line.note
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {"adjustment_note": raw}
    if isinstance(data, dict) and data.get("_v") == 1:
        return data
    return {"adjustment_note": raw}


def _set_line_note_state(line: PartnerInvoiceLine, state: dict[str, Any]) -> None:
    adjustment = state.get("adjustment_note")
    extra = {
        key: value
        for key, value in state.items()
        if key not in {"adjustment_note", "_v"} and value not in (None, "", {}, [])
    }
    if not extra:
        line.note = str(adjustment)[:500] if adjustment else None
        return
    payload: dict[str, Any] = {"_v": 1, **extra}
    if adjustment:
        payload["adjustment_note"] = str(adjustment)[:180]
    encoded = json.dumps(payload, ensure_ascii=False)
    if len(encoded) > 500:
        payload.pop("discount_note", None)
        payload.pop("adjustment_note", None)
        encoded = json.dumps(payload, ensure_ascii=False)[:500]
    line.note = encoded


def _ui_line_type(source_type: str | None) -> str:
    if source_type in {"custom", "manual_discount"}:
        return "custom"
    return "ride"


def _line_meta_from_state(
    state: dict[str, Any], *, service_date: str | None, source_type: str | None
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if service_date:
        meta["service_date"] = service_date
    if state.get("manual_discount") or source_type == "manual_discount":
        meta["manual_discount"] = True
    if isinstance(state.get("custom_prestation"), dict):
        meta["custom_prestation"] = state["custom_prestation"]
    if state.get("original_amount") is not None:
        meta["original_line_total"] = state["original_amount"]
    if state.get("discount_scope") == "global":
        meta["global_discount_applied"] = True
    if state.get("discount_scope") == "per_line":
        meta["per_line_discount_applied"] = True
    return meta


def enrich_partner_line_dict(
    payload: dict[str, Any],
    *,
    source_type: str | None,
    service_date: str | None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Ajoute le contrat UI commun sans retirer les champs natifs partenaire."""
    note_payload = state or {}
    adjustment = note_payload.get("adjustment_note")
    if adjustment is None and "_v" not in note_payload:
        adjustment = payload.get("note")
    line_meta = _line_meta_from_state(
        note_payload, service_date=service_date, source_type=source_type
    )
    amount = payload.get("amount")
    enriched = dict(payload)
    enriched["type"] = _ui_line_type(source_type)
    enriched["line_total"] = float(amount) if amount is not None else None
    enriched["adjustment_note"] = adjustment
    enriched["line_meta"] = line_meta or None
    enriched["pickup_label"] = pickup_display_label(
        payload.get("pickup_time_source") or SOURCE_UNKNOWN,
        payload.get("scheduled_pickup_at"),
        payload.get("boarded_at"),
    )
    if "_v" in note_payload or adjustment is not None:
        enriched["note"] = adjustment
    return enriched


def serialize_partner_invoice_lines(
    partner_invoice: PartnerInvoice,
) -> list[dict[str, Any]]:
    """Lignes persistées, ou snapshot mémoire depuis les transferts (GET sans écriture)."""
    if partner_invoice.lines:
        rows: list[dict[str, Any]] = []
        for line in partner_invoice.lines:
            rows.append(
                enrich_partner_line_dict(
                    line.to_dict(),
                    source_type=line.source_type,
                    service_date=line.service_date,
                    state=note_state(line),
                )
            )
        return rows
    transfers = load_partner_invoice_transfers(partner_invoice)
    rows = []
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
        rows.append(
            enrich_partner_line_dict(
                payload,
                source_type="booking_transfer",
                service_date=payload.get("service_date"),
            )
        )
    return rows


def partner_invoice_editor_meta(partner_invoice: PartnerInvoice) -> dict[str, Any]:
    """Méta remise reconstruite depuis l'état des lignes (pas de colonne meta)."""
    gross = Decimal("0")
    current = Decimal("0")
    percent: Any = None
    note: str | None = None
    per_line: list[dict[str, Any]] = []
    for line in partner_invoice.lines or []:
        state = note_state(line)
        scope = state.get("discount_scope")
        if scope == "global" and state.get("original_amount") is not None:
            gross += _as_decimal(state["original_amount"])
            current += _as_decimal(line.amount)
            percent = state.get("discount_percent", percent)
            note = state.get("discount_note") or note
        elif scope == "per_line" and line.id is not None:
            per_line.append(
                {
                    "line_id": line.id,
                    "percent": state.get("discount_percent"),
                }
            )
    meta: dict[str, Any] = {}
    if gross > 0:
        meta["global_discount"] = {
            "percent": float(percent) if percent is not None else None,
            "note": note,
            "subtotal_before_ht": float(round_to_5_cents(gross)),
            "amount_ht": float(round_to_5_cents(gross - current)),
        }
    if per_line:
        meta["per_line_discounts"] = {"lines": per_line}
    return meta


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
        if "amount" in raw or "line_total" in raw:
            raw_amount = raw["amount"] if "amount" in raw else raw.get("line_total")
            line.amount = round_to_5_cents(_as_decimal(raw_amount))
            if "unit_price" not in raw and line.quantity:
                line.unit_price = round_to_5_cents(line.amount / line.quantity)
        elif "quantity" in raw or "unit_price" in raw:
            line.amount = round_to_5_cents(line.quantity * line.unit_price)
        if "note" in raw or "adjustment_note" in raw:
            state = note_state(line)
            raw_note = (
                raw["adjustment_note"] if "adjustment_note" in raw else raw.get("note")
            )
            state["adjustment_note"] = str(raw_note)[:180] if raw_note else None
            _set_line_note_state(line, state)
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


def _discountable_partner_lines(
    partner_invoice: PartnerInvoice,
) -> list[PartnerInvoiceLine]:
    rows: list[PartnerInvoiceLine] = []
    for line in partner_invoice.lines or []:
        if line.source_type not in {"booking_transfer", "custom"}:
            continue
        if _as_decimal(line.amount) <= 0:
            continue
        rows.append(line)
    return rows


def restore_partner_percent_discounts(partner_invoice: PartnerInvoice) -> None:
    """Réaffiche les montants catalogue avant une nouvelle remise ou sa suppression."""
    for line in partner_invoice.lines or []:
        state = note_state(line)
        if state.get("discount_scope") not in {"global", "per_line"}:
            continue
        if state.get("original_amount") is not None:
            line.amount = round_to_5_cents(_as_decimal(state["original_amount"]))
        if state.get("original_unit_price") is not None:
            line.unit_price = round_to_5_cents(
                _as_decimal(state["original_unit_price"])
            )
        if state.get("original_quantity") is not None:
            line.quantity = round_to_5_cents(_as_decimal(state["original_quantity"]))
        for key in (
            "original_amount",
            "original_unit_price",
            "original_quantity",
            "discount_scope",
            "discount_percent",
            "discount_note",
        ):
            state.pop(key, None)
        _set_line_note_state(line, state)


def _apply_percent_on_line(
    line: PartnerInvoiceLine,
    *,
    percent: float,
    scope: str,
    note: str | None,
) -> None:
    state = note_state(line)
    if state.get("discount_scope") not in {"global", "per_line"}:
        state["original_amount"] = format(line.amount or Decimal("0"), "f")
        state["original_unit_price"] = format(line.unit_price or Decimal("0"), "f")
        state["original_quantity"] = format(line.quantity or Decimal("1"), "f")
    factor = (Decimal("100") - Decimal(str(percent))) / Decimal("100")
    line.amount = round_to_5_cents(_as_decimal(line.amount) * factor)
    if line.quantity and _as_decimal(line.quantity) != 0:
        line.unit_price = round_to_5_cents(line.amount / line.quantity)
    state["discount_scope"] = scope
    state["discount_percent"] = percent
    if note:
        state["discount_note"] = str(note)[:180]
    _set_line_note_state(line, state)


def apply_partner_global_discount(
    partner_invoice: PartnerInvoice, *, percent: float, note: str | None
) -> None:
    if not (0 < float(percent) <= 100):
        raise PartnerDraftEditError("Remise: pourcentage invalide (0-100]")
    restore_partner_percent_discounts(partner_invoice)
    eligible = _discountable_partner_lines(partner_invoice)
    if not eligible:
        raise PartnerDraftEditError(
            "Aucune ligne HT à remiser (transport ou prestation)."
        )
    for line in eligible:
        _apply_percent_on_line(line, percent=float(percent), scope="global", note=note)


def apply_partner_per_line_discounts(
    partner_invoice: PartnerInvoice, line_discounts: list[dict[str, Any]]
) -> None:
    if not line_discounts:
        raise PartnerDraftEditError(
            "Indiquez au moins un pourcentage sur une ligne remisable."
        )
    restore_partner_percent_discounts(partner_invoice)
    by_id = {line.id: line for line in partner_invoice.lines if line.id is not None}
    applied = 0
    for raw in line_discounts:
        try:
            line_id = int(raw["line_id"])
            percent = float(raw["percent"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PartnerDraftEditError("Remise par ligne invalide") from exc
        if not (0 < percent <= 100):
            raise PartnerDraftEditError("Chaque pourcentage doit être entre 0 et 100.")
        line = by_id.get(line_id)
        if line is None or line not in _discountable_partner_lines(partner_invoice):
            raise PartnerDraftEditError("Ligne non remisable.")
        _apply_percent_on_line(line, percent=percent, scope="per_line", note=None)
        applied += 1
    if applied == 0:
        raise PartnerDraftEditError("Aucune remise par ligne appliquée.")


def remove_partner_invoice_line(partner_invoice: PartnerInvoice, line_id: int) -> None:
    """Retire le snapshot. Un transfert source est détaché pour ne pas être resynthétisé."""
    target = next(
        (line for line in partner_invoice.lines if line.id == int(line_id)), None
    )
    if target is None:
        raise PartnerDraftEditError("Ligne introuvable", status_code=404)
    if target.source_type == "booking_transfer" and target.source_id is not None:
        linked = next(
            (
                transfer
                for transfer in list(partner_invoice.transfers or [])
                if transfer.id == target.source_id
            ),
            None,
        )
        if linked is not None:
            partner_invoice.transfers.remove(linked)
    partner_invoice.lines.remove(target)
    db.session.flush()


def add_partner_custom_line(
    partner_invoice: PartnerInvoice, payload: dict[str, Any]
) -> None:
    """Ligne HT personnalisée ou déduction libre. N'altère pas les transferts source."""
    desc = str(payload.get("description") or "").strip()[:500]
    if not desc:
        raise PartnerDraftEditError("Description requise")
    try:
        line_total = round_to_5_cents(_as_decimal(payload.get("line_total")))
    except Exception as exc:
        raise PartnerDraftEditError("Montant HT invalide") from exc
    if line_total == 0:
        raise PartnerDraftEditError("Le montant HT ne peut pas être zéro")
    is_manual = line_total < 0
    if is_manual:
        quantity = Decimal("1")
        unit = line_total
    else:
        try:
            quantity = _as_decimal(
                payload.get("qty") or payload.get("quantity") or "1", "1"
            )
        except Exception:
            quantity = Decimal("1")
        if quantity <= 0:
            quantity = Decimal("1")
        unit = round_to_5_cents(line_total / quantity)
    state: dict[str, Any] = {}
    if is_manual:
        state["manual_discount"] = True
    else:
        mode = payload.get("custom_mode")
        if mode in {"time", "quantity"}:
            entry: dict[str, Any] = {"mode": str(mode)}
            if mode == "time":
                unit_name = str(payload.get("time_unit") or "h")
                entry["time_unit"] = (
                    unit_name if unit_name in {"min", "h", "d", "mois"} else "h"
                )
            state["custom_prestation"] = entry
    service_date = payload.get("service_date_iso") or payload.get("service_date")
    sort_order = (
        max((line.sort_order or 0) for line in partner_invoice.lines) + 1
        if partner_invoice.lines
        else 0
    )
    line = PartnerInvoiceLine(
        description=desc,
        quantity=quantity,
        unit_price=unit,
        amount=line_total,
        source_type="manual_discount" if is_manual else "custom",
        source_id=None,
        sort_order=sort_order,
        service_date=str(service_date)[:20] if service_date else None,
        client_name=desc[:200],
        pickup_time_source=SOURCE_UNKNOWN,
    )
    active_global = next(
        (
            note_state(existing).get("discount_percent")
            for existing in partner_invoice.lines
            if note_state(existing).get("discount_scope") == "global"
        ),
        None,
    )
    partner_invoice.lines.append(line)
    db.session.flush()
    _set_line_note_state(line, state)
    if active_global is not None and not is_manual:
        _apply_percent_on_line(
            line,
            percent=float(active_global),
            scope="global",
            note=None,
        )
