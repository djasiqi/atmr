"""Édition minimale d’une facture brouillon (V1) : ligne, suppression, remise globale."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy.orm import joinedload

from ext import db
from infrastructure.invoices.invoice_calculator import InvoiceCalculator, round_to_5_cents
from models import Booking, Invoice, InvoiceLine, InvoiceStatus
from models.enums import InvoiceLineType
from repositories.company_billing_settings_repository import CompanyBillingSettingsRepository
from services.documents.pdf import PDFService

logger = logging.getLogger(__name__)


def _line_needs_ttc_repair(l: InvoiceLine) -> bool:
    """True si HT ≠ 0 mais TTC ligne absent ou 0 (données incohérentes)."""
    lt = l.line_total
    if lt is None:
        return False
    twi = l.total_with_vat
    return twi is None or (twi == 0 and lt != 0)


def repair_draft_invoice_if_line_totals_inconsistent(invoice: Invoice) -> bool:
    """Corrige les totaux ligne/facture pour les brouillons ayant des TTC ligne à 0 avec HT ≠ 0.

    Utile au GET pour afficher un total cohérent sans action utilisateur.
    Retourne True si une correction a été appliquée (commit attendu côté appelant).
    """
    if invoice.status != InvoiceStatus.DRAFT:
        return False
    lines = [l for l in invoice.lines if l is not None]
    if not any(_line_needs_ttc_repair(l) for l in lines):
        return False
    _recompute_totals_from_lines(invoice)
    return True


def _recompute_totals_from_lines(invoice: Invoice) -> None:
    """Recalcule sous-totaux, TVA et TTC facture à partir des lignes.

    Rattrapage : si `total_with_vat` vaut 0 (ou null) en base alors que
    `line_total` (et évent. TVA) ne le sont pas, la somme
    `sum(l.total_with_vat or 0)` excluait la ligne — ex. ligne CUSTOM
    enregistrée sans TTC cohérent. On recalcule le TTC ligne = HT + TVA.
    """
    lines = [l for l in invoice.lines if l is not None]
    for l in lines:
        if not _line_needs_ttc_repair(l):
            continue
        lt = l.line_total or Decimal("0")
        va = l.vat_amount or Decimal("0")
        l.total_with_vat = round_to_5_cents(lt + va)
    sub = sum((l.line_total or Decimal("0")) for l in lines)
    vat = sum((l.vat_amount or Decimal("0")) for l in lines)
    tw = sum((l.total_with_vat or Decimal("0")) for l in lines)
    invoice.subtotal_amount = round_to_5_cents(sub)
    invoice.vat_total_amount = round_to_5_cents(vat)
    invoice.total_amount = round_to_5_cents(tw)
    paid = invoice.amount_paid or Decimal("0")
    invoice.balance_due = round_to_5_cents(invoice.total_amount - paid)
    # vat_breakdown simplifié
    breakdown: dict[str, dict[str, float]] = {}
    for l in lines:
        if l.vat_rate is not None and l.line_total is not None:
            key = f"{float(l.vat_rate):.2f}"
            if key not in breakdown:
                breakdown[key] = {"base": 0.0, "vat": 0.0}
            breakdown[key]["base"] += float(l.line_total)
            breakdown[key]["vat"] += float(l.vat_amount or 0)
    if breakdown:
        invoice.vat_breakdown = breakdown  # type: ignore[assignment]


@dataclass(frozen=True, slots=True)
class EditDraftResult:
    success: bool
    invoice: Invoice | None = None
    error: dict[str, str] | None = None
    status_code: int | None = None


def _resolve_draft_invoice(company_id: int, invoice_id: int) -> tuple[Invoice | None, int | None, str | None]:
    """Charge la facture ; 404 si absente, message si non brouillon."""
    inv = (
        Invoice.query.options(joinedload(Invoice.lines))
        .filter_by(id=invoice_id, company_id=company_id)
        .first()
    )
    if not inv:
        return None, 404, "Facture introuvable"
    st = inv.status.value if hasattr(inv.status, "value") else str(inv.status)
    if st != "draft":
        return None, 400, "Seules les factures au statut brouillon peuvent être modifiées."
    return inv, None, None


def remove_draft_invoice_line(company_id: int, invoice_id: int, line_id: int) -> EditDraftResult:
    inv, err_code, err_msg = _resolve_draft_invoice(company_id, invoice_id)
    if not inv:
        return EditDraftResult(
            False,
            error={"error": err_msg or "Erreur"},
            status_code=err_code or 400,
        )
    logger.info(
        "[draft_edit] remove_line invoice_id=%s line_id=%s company_id=%s",
        invoice_id,
        line_id,
        company_id,
    )

    line = next((l for l in inv.lines if l.id == line_id), None)
    if not line:
        return EditDraftResult(False, error={"error": "Ligne introuvable"}, status_code=404)

    if line.reservation_id:
        bk = Booking.query.get(line.reservation_id)
        if bk and getattr(bk, "invoice_line_id", None) == line.id:
            bk.invoice_line_id = None

    db.session.delete(line)
    db.session.flush()
    db.session.expire(inv, ["lines"])
    _recompute_totals_from_lines(inv)
    try:
        pdf = PDFService()
        inv.pdf_url = pdf.generate_invoice_pdf(inv)
    except Exception as e:
        logger.warning("PDF draft après suppression ligne: %s", e)
    db.session.commit()
    return EditDraftResult(True, invoice=inv)


def update_draft_invoice_line(
    company_id: int,
    invoice_id: int,
    line_id: int,
    *,
    line_total: float | None = None,
    adjustment_note: str | None = None,
) -> EditDraftResult:
    inv, err_code, err_msg = _resolve_draft_invoice(company_id, invoice_id)
    if not inv:
        return EditDraftResult(
            False,
            error={"error": err_msg or "Erreur"},
            status_code=err_code or 400,
        )
    logger.info(
        "[draft_edit] update_line invoice_id=%s line_id=%s line_total=%s company_id=%s",
        invoice_id,
        line_id,
        line_total,
        company_id,
    )

    line = next((l for l in inv.lines if l.id == line_id), None)
    if not line:
        return EditDraftResult(False, error={"error": "Ligne introuvable"}, status_code=404)

    settings_repo = CompanyBillingSettingsRepository()
    bset = settings_repo.find_or_create(company_id)
    vat_rate = Decimal(str(bset.vat_rate)) if bset.vat_rate is not None else Decimal("0.00")
    vat_applicable = bool(bset.vat_applicable) and vat_rate > 0

    if line_total is not None:
        lt = round_to_5_cents(Decimal(str(line_total)))
        line.line_total = lt
        line.unit_price = lt / (line.qty or Decimal("1"))
        calc = InvoiceCalculator()
        if vat_applicable and vat_rate > 0:
            line.vat_rate = vat_rate
            va, tw = calc.calculate_vat(lt, vat_rate)
            line.vat_amount = va
            line.total_with_vat = tw
        else:
            line.vat_rate = None
            line.vat_amount = Decimal("0.00")
            line.total_with_vat = lt

    if adjustment_note is not None:
        line.adjustment_note = adjustment_note[:2000] if adjustment_note else None

    _recompute_totals_from_lines(inv)
    try:
        pdf = PDFService()
        inv.pdf_url = pdf.generate_invoice_pdf(inv)
    except Exception as e:
        logger.warning("PDF draft après MAJ ligne: %s", e)
    db.session.commit()
    return EditDraftResult(True, invoice=inv)


def _restore_ride_amounts_from_bookings(inv: Invoice) -> None:
    """Reprend le montant HT depuis la réservation (évite de remiser deux fois sur un net déjà remis)."""
    calc = InvoiceCalculator()
    settings_repo = CompanyBillingSettingsRepository()
    bset = settings_repo.find_or_create(inv.company_id)
    vat_rate = bset.vat_rate or Decimal("0.00")
    vat_applicable = bset.vat_applicable and (vat_rate > 0)

    for lm in inv.lines:
        if lm.type != InvoiceLineType.RIDE or not lm.reservation_id:
            continue
        bk = Booking.query.get(lm.reservation_id)
        if not bk:
            continue
        raw = getattr(bk, "amount", None) or getattr(bk, "estimated_amount", None) or 0
        try:
            lt = round_to_5_cents(Decimal(str(raw)))
        except Exception:
            lt = Decimal("0.00")
        lm.line_total = lt
        lm.unit_price = lt / (lm.qty or Decimal("1"))
        if vat_applicable:
            lm.vat_rate = vat_rate
            va, tw = calc.calculate_vat(lt, vat_rate)
            lm.vat_amount = va
            lm.total_with_vat = tw
        else:
            lm.vat_rate = None
            lm.vat_amount = Decimal("0.00")
            lm.total_with_vat = lt


def _format_percent_label(gd: Decimal) -> str:
    """Libellé pourcentage sans notation scientifique (ex. 20, pas 2E+1)."""
    try:
        f = float(gd)
    except Exception:
        return str(gd).strip()
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{f:.4f}".rstrip("0").rstrip(".") or "0"


def apply_draft_global_discount(
    company_id: int,
    invoice_id: int,
    *,
    global_discount_percent: float,
    global_discount_note: str | None = None,
) -> EditDraftResult:
    """Applique une remise globale sur les lignes RIDE : ligne CUSTOM négative + recompute HT/TVA (aligné generate_invoice)."""
    inv, err_code, err_msg = _resolve_draft_invoice(company_id, invoice_id)
    if not inv:
        return EditDraftResult(
            False,
            error={"error": err_msg or "Erreur"},
            status_code=err_code or 400,
        )
    logger.info(
        "[draft_edit] apply_global_discount invoice_id=%s percent=%s company_id=%s",
        invoice_id,
        global_discount_percent,
        company_id,
    )

    if not (0 < global_discount_percent <= 100):
        return EditDraftResult(
            False, error={"error": "Remise: pourcentage invalide (0-100]"}, status_code=400
        )

    # Retirer les anciennes lignes de remise CUSTOM (négatif, pas de réservation)
    neg_custom = InvoiceLine.query.filter(
        InvoiceLine.invoice_id == inv.id,
        InvoiceLine.type == InvoiceLineType.CUSTOM,
        InvoiceLine.reservation_id.is_(None),
        InvoiceLine.line_total < 0,
    ).all()
    for l in neg_custom:
        db.session.delete(l)
    db.session.flush()
    db.session.expire(inv, ["lines"])
    # Repartir des montants « catalogue » issus des réservations
    _restore_ride_amounts_from_bookings(inv)
    db.session.flush()

    ride_lines = [l for l in inv.lines if l.type == InvoiceLineType.RIDE]
    gross = sum((x.line_total or Decimal("0")) for x in ride_lines)
    if gross <= 0:
        return EditDraftResult(
            False, error={"error": "Aucun montant HT (lignes transport) à remiser"}, status_code=400
        )

    gd = Decimal(str(global_discount_percent))
    discount_ht = round_to_5_cents(gross * gd / Decimal("100"))
    if discount_ht > gross:
        discount_ht = gross
    net_ht = round_to_5_cents(gross - discount_ht)

    note = (global_discount_note or "").strip()
    pct_lbl = _format_percent_label(gd)
    desc = f"Remise commerciale {pct_lbl} %"
    if note:
        desc = f"{desc} — {note[:300]}"

    # Repartir le net HT sur les RIDE (dernière ligne ajuste le reste) — aligné generate_invoice
    running = Decimal("0.00")
    n = len(ride_lines)
    settings_repo = CompanyBillingSettingsRepository()
    bset = settings_repo.find_or_create(company_id)
    vat_rate = Decimal(str(bset.vat_rate)) if bset.vat_rate is not None else Decimal("0.00")
    vat_applicable = bool(bset.vat_applicable) and vat_rate > 0
    calc = InvoiceCalculator()

    for idx, lm in enumerate(ride_lines):
        base = lm.line_total or Decimal("0")
        if idx < n - 1:
            part = round_to_5_cents(base * net_ht / gross) if gross else Decimal("0.00")
        else:
            part = round_to_5_cents(net_ht - running)
        running += part
        lm.line_total = part
        lm.unit_price = part / (lm.qty or Decimal("1"))
        if vat_applicable and vat_rate > 0:
            lm.vat_rate = vat_rate
            va, tw = calc.calculate_vat(part, vat_rate)
            lm.vat_amount = va
            lm.total_with_vat = tw
        else:
            lm.vat_rate = None
            lm.vat_amount = Decimal("0.00")
            lm.total_with_vat = part

    db.session.add(
        InvoiceLine(
            invoice_id=inv.id,
            type=InvoiceLineType.CUSTOM,
            description=desc[:500],
            qty=Decimal("1"),
            unit_price=-discount_ht,
            line_total=-discount_ht,
            vat_rate=None,
            vat_amount=Decimal("0.00"),
            total_with_vat=-discount_ht,
            adjustment_note=None,
            reservation_id=None,
        )
    )
    db.session.flush()
    _recompute_totals_from_lines(inv)
    meta = dict(inv.meta) if isinstance(inv.meta, dict) else {}
    meta["global_discount"] = {
        "percent": float(gd),
        "amount_ht": float(discount_ht),
        "subtotal_before_ht": float(gross),
    }
    if note:
        meta["global_discount"]["note"] = note
    inv.meta = meta
    try:
        pdf = PDFService()
        inv.pdf_url = pdf.generate_invoice_pdf(inv)
    except Exception as e:
        logger.warning("PDF draft après remise: %s", e)
    db.session.commit()
    return EditDraftResult(True, invoice=inv)


def remove_draft_global_discount(company_id: int, invoice_id: int) -> EditDraftResult:
    """Supprime la remise globale : restaure les montants RIDE catalogue, retire meta et la ligne négative."""
    inv, err_code, err_msg = _resolve_draft_invoice(company_id, invoice_id)
    if not inv:
        return EditDraftResult(
            False,
            error={"error": err_msg or "Erreur"},
            status_code=err_code or 400,
        )
    logger.info(
        "[draft_edit] remove_global_discount invoice_id=%s company_id=%s",
        invoice_id,
        company_id,
    )

    # Toute ligne CUSTOM négative sans réservation = remise globale (y compris libellé « 2E+1 % »
    # ou autre) — ne pas se fier au texte « remise » ni au cache ORM seul.
    neg_custom = InvoiceLine.query.filter(
        InvoiceLine.invoice_id == inv.id,
        InvoiceLine.type == InvoiceLineType.CUSTOM,
        InvoiceLine.reservation_id.is_(None),
        InvoiceLine.line_total < 0,
    ).all()
    for l in neg_custom:
        db.session.delete(l)
    db.session.flush()
    db.session.expire(inv, ["lines"])
    # Toujours restaurer les RIDE (catalogue) et retirer la trace meta, même si la ligne négative manquait.
    _restore_ride_amounts_from_bookings(inv)
    db.session.flush()
    _recompute_totals_from_lines(inv)
    meta = dict(inv.meta) if isinstance(inv.meta, dict) else {}
    meta.pop("global_discount", None)
    inv.meta = meta if meta else None
    try:
        pdf = PDFService()
        inv.pdf_url = pdf.generate_invoice_pdf(inv)
    except Exception as e:
        logger.warning("PDF draft après annulation remise: %s", e)
    db.session.commit()
    return EditDraftResult(True, invoice=inv)


def add_draft_custom_line(
    company_id: int,
    invoice_id: int,
    *,
    description: str,
    line_total: float,
    qty: float = 1.0,
    custom_mode: str | None = None,
    time_unit: str | None = None,
) -> EditDraftResult:
    """Ajoute une ligne HT personnalisée (ex. accompagnement, forfait) — type CUSTOM, sans réservation."""
    inv, err_code, err_msg = _resolve_draft_invoice(company_id, invoice_id)
    if not inv:
        return EditDraftResult(
            False,
            error={"error": err_msg or "Erreur"},
            status_code=err_code or 400,
        )
    desc = (description or "").strip()[:500]
    if not desc:
        return EditDraftResult(
            False, error={"error": "Description requise"}, status_code=400
        )
    try:
        lt = round_to_5_cents(Decimal(str(line_total)))
    except Exception:
        return EditDraftResult(
            False, error={"error": "Montant HT invalide"}, status_code=400
        )
    if lt <= 0:
        return EditDraftResult(
            False,
            error={"error": "Le montant HT doit être strictement positif"},
            status_code=400,
        )
    try:
        q = Decimal(str(qty))
    except Exception:
        q = Decimal("1")
    if q <= 0:
        q = Decimal("1")
    unit = round_to_5_cents(lt / q) if q else lt

    line_meta: dict | None = None
    if custom_mode in ("time", "quantity"):
        entry: dict[str, Any] = {"mode": str(custom_mode)}
        if custom_mode == "time" and time_unit in ("min", "h", "d", "mois"):
            entry["time_unit"] = str(time_unit)
        line_meta = {"custom_prestation": entry}

    logger.info(
        "[draft_edit] add_custom_line invoice_id=%s company_id=%s",
        invoice_id,
        company_id,
    )
    settings_repo = CompanyBillingSettingsRepository()
    bset = settings_repo.find_or_create(company_id)
    vat_rate = Decimal(str(bset.vat_rate)) if bset.vat_rate is not None else Decimal("0.00")
    vat_applicable = bool(bset.vat_applicable) and vat_rate > 0
    calc = InvoiceCalculator()

    line = InvoiceLine(
        invoice_id=inv.id,
        type=InvoiceLineType.CUSTOM,
        description=desc,
        qty=q,
        unit_price=unit,
        line_total=lt,
        vat_rate=None,
        vat_amount=Decimal("0.00"),
        total_with_vat=lt,
        adjustment_note=None,
        reservation_id=None,
        line_meta=line_meta,
    )
    if vat_applicable and vat_rate > 0:
        line.vat_rate = vat_rate
        va, tw = calc.calculate_vat(lt, vat_rate)
        line.vat_amount = va
        line.total_with_vat = tw
    db.session.add(line)
    db.session.flush()
    _recompute_totals_from_lines(inv)
    try:
        pdf = PDFService()
        inv.pdf_url = pdf.generate_invoice_pdf(inv)
    except Exception as e:
        logger.warning("PDF draft après ligne perso: %s", e)
    db.session.commit()
    return EditDraftResult(True, invoice=inv)
