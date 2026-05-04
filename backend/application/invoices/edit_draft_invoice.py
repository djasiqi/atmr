"""Édition minimale des lignes facture (brouillon ou déjà émise).

Statuts éditables : brouillon, envoyée, partiellement payée, en retard — pas payée ni annulée.

Le PDF n'est pas régénéré sur ces mutations : l'éditeur utilise un aperçu HTML ; le fichier PDF est
produit à la finalisation / envoi / téléchargement ou via ``POST …/regenerate-pdf``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, cast

from sqlalchemy.orm import joinedload

from ext import db
from infrastructure.invoices.invoice_calculator import (
    InvoiceCalculator,
    round_to_5_cents,
)
from models import Booking, Invoice, InvoiceLine, InvoiceStatus
from models.enums import InvoiceLineType
from repositories.company_billing_settings_repository import (
    CompanyBillingSettingsRepository,
)

logger = logging.getLogger(__name__)

_EDITABLE_INVOICE_LINE_STATUSES: frozenset[InvoiceStatus] = frozenset(
    {
        InvoiceStatus.DRAFT,
        InvoiceStatus.SENT,
        InvoiceStatus.PARTIALLY_PAID,
        InvoiceStatus.OVERDUE,
    }
)


def invoice_allows_line_editing(inv: Invoice) -> bool:
    """True si lignes / remises peuvent être modifiées (pas payée, pas annulée)."""
    return inv.status in _EDITABLE_INVOICE_LINE_STATUSES


def _mark_pdf_stale(invoice: Invoice) -> None:
    """Marque le PDF comme obsolète (import paresseux pour éviter un cycle avec ``services.documents.pdf``)."""
    from application.invoices.invoice_pdf_state import mark_pdf_stale

    mark_pdf_stale(invoice)


# Ligne CUSTOM < 0 ajoutée explicitement comme remise libre (libellé + montant) — ne pas la traiter comme la ligne « remise globale ».
_META_MANUAL_DISCOUNT = "manual_discount"
_META_PER_LINE_DISCOUNT_LINE = "per_line_discount_line"
# Sauvegarde catalogue avant remise globale % (retrait si snapshot incomplet / migration).
_META_ORIGINAL_LINE_TOTAL = "original_line_total"
_ISO_DATE_SLICE_LEN = 10
_MAX_DISCOUNT_PERCENT = 100.0


def _normalize_optional_service_date_iso(raw: Any) -> str | None:
    """Accepte une date ``YYYY-MM-DD`` (corps POST ou extrait d'un ISO) ; sinon None."""
    if raw is None:
        return None
    s = str(raw).strip()[:_ISO_DATE_SLICE_LEN]
    if len(s) != _ISO_DATE_SLICE_LEN:
        return None
    try:
        return date.fromisoformat(s).isoformat()
    except ValueError:
        return None


def _is_manual_discount_line(line: InvoiceLine) -> bool:
    if line.type != InvoiceLineType.CUSTOM:
        return False
    meta = line.line_meta if isinstance(line.line_meta, dict) else {}
    return bool(meta.get(_META_MANUAL_DISCOUNT))


def _delete_negative_custom_discount_lines_except_manual(inv: Invoice) -> None:
    """Supprime les lignes CUSTOM négatives liées aux remises % auto (globale ou par ligne)."""
    neg_custom = InvoiceLine.query.filter(
        InvoiceLine.invoice_id == inv.id,
        InvoiceLine.type == InvoiceLineType.CUSTOM,
        InvoiceLine.reservation_id.is_(None),
        InvoiceLine.line_total < 0,
    ).all()
    for neg_ln in neg_custom:
        if _is_manual_discount_line(neg_ln):
            continue
        db.session.delete(neg_ln)
    db.session.flush()
    db.session.expire(inv, ["lines"])


_DRAFT_TTC_REPAIR_EPS = Decimal("0.02")
_EXPECTED_UPDATED_AT_TOLERANCE_SEC = 2.0


def _error_payload_from_resolve(
    err_msg: str | dict[str, Any] | None, default: str = "Erreur"
) -> dict[str, str]:
    if isinstance(err_msg, dict):
        return err_msg
    return {"error": str(err_msg or default)}


def _invoice_updated_at_utc(inv: Invoice) -> datetime | None:
    u = cast(datetime, inv.updated_at)
    if u.tzinfo is None:
        return u.replace(tzinfo=UTC)
    return u.astimezone(UTC)


def _parse_iso_datetime_utc(raw: object) -> datetime | None:
    if raw is None:
        return None
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _canonical_line_total_with_vat(line: InvoiceLine) -> Decimal:
    """TTC ligne attendu : HT + TVA (arrondi 5 ct), source de vérité pour les brouillons."""
    lt = line.line_total or Decimal("0")
    va = line.vat_amount or Decimal("0")
    return round_to_5_cents(lt + va)


def _line_needs_ttc_repair(line: InvoiceLine) -> bool:
    """True si le TTC ligne en base ne correspond pas à HT + TVA (bug / données anciennes)."""
    canonical = _canonical_line_total_with_vat(line)
    twi = line.total_with_vat
    return abs(twi - canonical) > _DRAFT_TTC_REPAIR_EPS


def repair_draft_invoice_if_line_totals_inconsistent(invoice: Invoice) -> bool:
    """Recalcule totaux facture / lignes pour les brouillons lorsque TTC ≠ HT+TVA ou total facture ≠ Σ lignes.

    Utile au GET (détail ou liste) pour afficher le même montant que la somme des lignes.
    Retourne True si une correction a été appliquée (commit attendu côté appelant).
    """
    if not invoice_allows_line_editing(invoice):
        return False
    lines = [ln for ln in invoice.lines if ln is not None]
    if not lines:
        return False
    canonical_invoice_tw = round_to_5_cents(
        sum(
            (_canonical_line_total_with_vat(ln) for ln in lines),
            start=Decimal("0"),
        )
    )
    cur_total = invoice.total_amount or Decimal("0")
    mismatch_invoice = abs(cur_total - canonical_invoice_tw) > _DRAFT_TTC_REPAIR_EPS
    mismatch_lines = any(_line_needs_ttc_repair(ln) for ln in lines)
    if not mismatch_invoice and not mismatch_lines:
        return False
    _recompute_totals_from_lines(invoice)
    return True


def _recompute_totals_from_lines(invoice: Invoice) -> None:
    """Recalcule sous-totaux, TVA et TTC facture à partir des lignes.

    Le TTC de chaque ligne est toujours **HT + TVA** (arrondi), puis la facture agrège.
    Évite les dérives (ex. total_with_vat aberrant) qui désynchronisaient liste / PDF vs éditeur.
    """
    lines = [ln for ln in invoice.lines if ln is not None]
    for ln in lines:
        ln.total_with_vat = _canonical_line_total_with_vat(ln)
    sub = sum(
        (ln.line_total or Decimal("0") for ln in lines),
        start=Decimal("0"),
    )
    vat = sum(
        (ln.vat_amount or Decimal("0") for ln in lines),
        start=Decimal("0"),
    )
    tw = sum(
        (ln.total_with_vat or Decimal("0") for ln in lines),
        start=Decimal("0"),
    )
    invoice.subtotal_amount = round_to_5_cents(sub)
    invoice.vat_total_amount = round_to_5_cents(vat)
    invoice.total_amount = round_to_5_cents(tw)
    paid = invoice.amount_paid or Decimal("0")
    invoice.balance_due = round_to_5_cents(invoice.total_amount - paid)
    # vat_breakdown simplifié
    breakdown: dict[str, dict[str, float]] = {}
    for ln in lines:
        if ln.vat_rate is not None and ln.line_total is not None:
            key = f"{float(ln.vat_rate):.2f}"
            if key not in breakdown:
                breakdown[key] = {"base": 0.0, "vat": 0.0}
            breakdown[key]["base"] += float(ln.line_total)
            breakdown[key]["vat"] += float(ln.vat_amount or 0)
    if breakdown:
        invoice.vat_breakdown = breakdown


@dataclass(frozen=True, slots=True)
class EditDraftResult:
    success: bool
    invoice: Invoice | None = None
    error: dict[str, Any] | None = None
    status_code: int | None = None


def _expected_updated_at_conflict(
    inv: Invoice, expected_updated_at: str | None
) -> EditDraftResult | None:
    """409 si le client envoie un horodatage qui ne correspond pas à ``invoice.updated_at``."""
    if expected_updated_at is None:
        return None
    if not expected_updated_at.strip():
        return None
    exp = _parse_iso_datetime_utc(expected_updated_at)
    if exp is None:
        return EditDraftResult(
            False,
            error={"error": "expected_updated_at invalide (format ISO 8601 attendu)."},
            status_code=400,
        )
    cur = _invoice_updated_at_utc(inv)
    if cur is None:
        return None
    if abs((exp - cur).total_seconds()) > _EXPECTED_UPDATED_AT_TOLERANCE_SEC:
        return EditDraftResult(
            False,
            error={
                "error": (
                    "La facture a été modifiée ailleurs. Rechargez la page et réessayez."
                ),
                "error_code": "INVOICE_CONCURRENT_MODIFICATION",
            },
            status_code=409,
        )
    return None


def _resolve_draft_invoice(
    company_id: int,
    invoice_id: int,
    *,
    expected_updated_at: str | None = None,
) -> tuple[Invoice | None, int | None, str | dict[str, Any] | None]:
    """Charge la facture ; 404 si absente, message si non brouillon.

    Le verrou optimiste est vérifié **avant** le repair TTC pour que l'horodatage du GET reste valide.
    """
    inv = (
        Invoice.query.options(joinedload(Invoice.lines))
        .filter_by(id=invoice_id, company_id=company_id)
        .first()
    )
    if not inv:
        return None, 404, "Facture introuvable"
    if not invoice_allows_line_editing(inv):
        return (
            None,
            400,
            "Seules les factures non payées et non annulées (brouillon, envoyée, en cours d'encaissement) peuvent être modifiées.",
        )
    conflict = _expected_updated_at_conflict(inv, expected_updated_at)
    if conflict:
        return None, conflict.status_code or 409, conflict.error or {"error": "Conflit"}
    if repair_draft_invoice_if_line_totals_inconsistent(inv):
        db.session.flush()
        db.session.refresh(inv)
    return inv, None, None


def remove_draft_invoice_line(
    company_id: int,
    invoice_id: int,
    line_id: int,
    *,
    expected_updated_at: str | None = None,
) -> EditDraftResult:
    inv, err_code, err_msg = _resolve_draft_invoice(
        company_id, invoice_id, expected_updated_at=expected_updated_at
    )
    if not inv:
        return EditDraftResult(
            False,
            error=_error_payload_from_resolve(err_msg),
            status_code=err_code or 400,
        )
    logger.info(
        "[draft_edit] remove_line invoice_id=%s line_id=%s company_id=%s",
        invoice_id,
        line_id,
        company_id,
    )

    line = next((ln for ln in inv.lines if ln.id == line_id), None)
    if not line:
        return EditDraftResult(
            False, error={"error": "Ligne introuvable"}, status_code=404
        )

    if line.reservation_id:
        bk = Booking.query.get(line.reservation_id)
        if bk and getattr(bk, "invoice_line_id", None) == line.id:
            bk.invoice_line_id = None

    db.session.delete(line)
    db.session.flush()
    db.session.expire(inv, ["lines"])
    _recompute_totals_from_lines(inv)
    _mark_pdf_stale(inv)
    db.session.commit()
    return EditDraftResult(True, invoice=inv)


def update_draft_invoice_line(
    company_id: int,
    invoice_id: int,
    line_id: int,
    *,
    line_total: float | None = None,
    adjustment_note: str | None = None,
    description: str | None = None,
    expected_updated_at: str | None = None,
) -> EditDraftResult:
    inv, err_code, err_msg = _resolve_draft_invoice(
        company_id, invoice_id, expected_updated_at=expected_updated_at
    )
    if not inv:
        return EditDraftResult(
            False,
            error=_error_payload_from_resolve(err_msg),
            status_code=err_code or 400,
        )
    logger.info(
        "[draft_edit] update_line invoice_id=%s line_id=%s line_total=%s company_id=%s",
        invoice_id,
        line_id,
        line_total,
        company_id,
    )

    line = next((ln for ln in inv.lines if ln.id == line_id), None)
    if not line:
        return EditDraftResult(
            False, error={"error": "Ligne introuvable"}, status_code=404
        )

    settings_repo = CompanyBillingSettingsRepository()
    bset = settings_repo.find_or_create(company_id)
    vat_rate = (
        Decimal(str(bset.vat_rate)) if bset.vat_rate is not None else Decimal("0.00")
    )
    vat_applicable = bool(bset.vat_applicable) and vat_rate > 0

    if line_total is not None:
        lt = round_to_5_cents(Decimal(str(line_total)))
        line.line_total = lt
        line.unit_price = lt / (line.qty or Decimal("1"))
        calc = InvoiceCalculator()
        custom_neg = line.type == InvoiceLineType.CUSTOM and lt < 0
        gd_line = isinstance(line.line_meta, dict) and (
            line.line_meta.get("global_discount_line")
            or line.line_meta.get(_META_PER_LINE_DISCOUNT_LINE)
        )
        if vat_applicable and vat_rate > 0 and not custom_neg:
            line.vat_rate = vat_rate
            va, tw = calc.calculate_vat(lt, vat_rate)
            line.vat_amount = va
            line.total_with_vat = tw
        else:
            line.vat_rate = None
            line.vat_amount = Decimal("0.00")
            line.total_with_vat = lt
        meta_amt = dict(line.line_meta) if line.line_meta else {}
        meta_amt["amount_overridden"] = True
        if line.type == InvoiceLineType.CUSTOM and not gd_line:
            if lt < 0:
                meta_amt[_META_MANUAL_DISCOUNT] = True
            elif _META_MANUAL_DISCOUNT in meta_amt:
                del meta_amt[_META_MANUAL_DISCOUNT]
        line.line_meta = meta_amt

    if adjustment_note is not None:
        line.adjustment_note = adjustment_note[:2000] if adjustment_note else None

    if description is not None:
        line.description = str(description).strip()[:500]
        meta = dict(line.line_meta) if line.line_meta else {}
        meta["description_overridden"] = True
        line.line_meta = meta

    _recompute_totals_from_lines(inv)
    _mark_pdf_stale(inv)
    db.session.commit()
    return EditDraftResult(True, invoice=inv)


def _restore_ride_amounts_from_bookings(inv: Invoice) -> None:
    """Reprend le montant HT depuis la réservation (évite de remiser deux fois sur un net déjà remis)."""
    calc = InvoiceCalculator()
    settings_repo = CompanyBillingSettingsRepository()
    bset = settings_repo.find_or_create(inv.company_id)
    vat_rate = bset.vat_rate or Decimal("0.00")
    vat_applicable = bool(bset.vat_applicable) and (vat_rate > 0)

    for lm in inv.lines:
        if lm.type != InvoiceLineType.RIDE or not lm.reservation_id:
            continue
        bk = Booking.query.get(lm.reservation_id)
        if not bk:
            continue
        amt = getattr(bk, "amount", None)
        est = getattr(bk, "estimated_amount", None)
        raw = amt if amt is not None else (est if est is not None else 0)
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
        meta_lm = dict(lm.line_meta) if isinstance(lm.line_meta, dict) else {}
        meta_lm.pop(_META_ORIGINAL_LINE_TOTAL, None)
        meta_lm.pop("per_line_discount_percent", None)
        lm.line_meta = meta_lm if meta_lm else None


def _restore_positive_ride_lines_from_catalog_meta(
    inv: Invoice, company_id: int
) -> EditDraftResult | None:
    """Restaure les transports HT depuis `original_line_total` (remise globale % ou par ligne)."""
    settings_repo = CompanyBillingSettingsRepository()
    bset = settings_repo.find_or_create(company_id)
    cr_vat = (
        Decimal(str(bset.vat_rate)) if bset.vat_rate is not None else Decimal("0.00")
    )
    vat_ok = bool(bset.vat_applicable) and cr_vat > 0
    calc = InvoiceCalculator()
    for lm in inv.lines:
        if lm.type != InvoiceLineType.RIDE:
            continue
        lt = lm.line_total or Decimal("0")
        if lt <= 0:
            continue
        meta = lm.line_meta if isinstance(lm.line_meta, dict) else {}
        raw_ot = meta.get(_META_ORIGINAL_LINE_TOTAL)
        if raw_ot is None:
            continue
        try:
            restored = round_to_5_cents(Decimal(str(raw_ot)))
        except Exception:
            return EditDraftResult(
                False,
                error={
                    "error": (
                        "Retrait remise : montant catalogue invalide pour une ligne transport."
                    )
                },
                status_code=409,
            )
        lm.line_total = restored
        lm.unit_price = restored / (lm.qty or Decimal("1"))
        if vat_ok:
            lm.vat_rate = cr_vat
            va, tw = calc.calculate_vat(restored, cr_vat)
            lm.vat_amount = va
            lm.total_with_vat = tw
        else:
            lm.vat_rate = None
            lm.vat_amount = Decimal("0.00")
            lm.total_with_vat = restored
        meta2 = dict(lm.line_meta) if isinstance(lm.line_meta, dict) else {}
        meta2.pop(_META_ORIGINAL_LINE_TOTAL, None)
        meta2.pop("per_line_discount_percent", None)
        lm.line_meta = meta2 if meta2 else None
    return None


def _line_ht_snapshot_dict(inv_line: InvoiceLine) -> dict[str, Any]:
    """Sérialise une ligne pour restauration après retrait de la remise globale %."""
    vr = inv_line.vat_rate
    lt = getattr(inv_line.type, "value", None) or str(inv_line.type)
    return {
        "id": inv_line.id,
        "line_total": format(inv_line.line_total or Decimal("0"), "f"),
        "unit_price": format(inv_line.unit_price or Decimal("0"), "f"),
        "vat_amount": format(inv_line.vat_amount or Decimal("0"), "f"),
        "total_with_vat": format(inv_line.total_with_vat or Decimal("0"), "f"),
        "vat_rate": format(vr, "f") if vr is not None else None,
        "line_type": str(lt),
    }


def _apply_line_ht_snapshot(inv_line: InvoiceLine, snap: dict[str, Any]) -> None:
    inv_line.line_total = Decimal(str(snap["line_total"]))
    inv_line.unit_price = Decimal(str(snap["unit_price"]))
    inv_line.vat_amount = Decimal(str(snap.get("vat_amount") or "0"))
    inv_line.total_with_vat = Decimal(str(snap.get("total_with_vat") or "0"))
    vr_raw = snap.get("vat_rate")
    if vr_raw is None or vr_raw == "":
        inv_line.vat_rate = None
    else:
        inv_line.vat_rate = Decimal(str(vr_raw))
    meta = dict(inv_line.line_meta) if isinstance(inv_line.line_meta, dict) else {}
    meta.pop(_META_ORIGINAL_LINE_TOTAL, None)
    inv_line.line_meta = meta if meta else None


def _line_snapshots_from_global_discount_meta(
    inv: Invoice,
) -> list[dict[str, Any]] | None:
    meta = inv.meta if isinstance(inv.meta, dict) else {}
    gd = meta.get("global_discount")
    if not isinstance(gd, dict):
        return None
    snaps = gd.get("line_snapshots")
    if isinstance(snaps, list) and snaps:
        return snaps
    return None


def _restore_invoice_lines_from_global_discount_snapshots(
    inv: Invoice, snapshots: list[dict[str, Any]]
) -> None:
    by_id: dict[int, dict[str, Any]] = {}
    for s in snapshots:
        try:
            by_id[int(s["id"])] = s
        except (KeyError, TypeError, ValueError):
            continue
    for lm in inv.lines:
        if lm.id is None:
            continue
        snap = by_id.get(lm.id)
        if snap:
            _apply_line_ht_snapshot(lm, snap)


def _eligible_positive_custom_lines_for_global_discount(
    inv: Invoice,
) -> list[InvoiceLine]:
    """Prestations CUSTOM HT > 0 (hors remises auto, déductions manuelles, lignes techniques)."""
    res: list[InvoiceLine] = []
    for inv_ln in inv.lines:
        if inv_ln.type != InvoiceLineType.CUSTOM:
            continue
        lt = inv_ln.line_total or Decimal("0")
        if lt <= 0:
            continue
        meta = inv_ln.line_meta if isinstance(inv_ln.line_meta, dict) else {}
        if meta.get("global_discount_line") or meta.get(_META_PER_LINE_DISCOUNT_LINE):
            continue
        if _is_manual_discount_line(inv_ln):
            continue
        res.append(inv_ln)
    return res


def _eligible_lines_for_per_line_discount(inv: Invoice) -> list[InvoiceLine]:
    """Lignes positives remisable individuellement : transports + prestations CUSTOM positives."""
    res: list[InvoiceLine] = []
    for inv_ln in inv.lines:
        lt = inv_ln.line_total or Decimal("0")
        if lt <= 0:
            continue
        if inv_ln.type == InvoiceLineType.RIDE:
            res.append(inv_ln)
            continue
        if inv_ln.type == InvoiceLineType.CUSTOM:
            meta = inv_ln.line_meta if isinstance(inv_ln.line_meta, dict) else {}
            if meta.get("global_discount_line") or meta.get(
                _META_PER_LINE_DISCOUNT_LINE
            ):
                continue
            if _is_manual_discount_line(inv_ln):
                continue
            res.append(inv_ln)
    return res


def _undo_auto_discount_lines_and_restore_catalog_for_global_discount(
    inv: Invoice,
) -> None:
    """Supprime les lignes CUSTOM de remise auto et rétablit les HT avant remise globale (snapshot ou réservations)."""
    prev_snaps = _line_snapshots_from_global_discount_meta(inv)
    _delete_negative_custom_discount_lines_except_manual(inv)
    if prev_snaps:
        _restore_invoice_lines_from_global_discount_snapshots(inv, prev_snaps)
    else:
        _restore_ride_amounts_from_bookings(inv)
    db.session.flush()
    db.session.expire(inv, ["lines"])


def _stash_original_line_totals_for_global_discount(
    eligible: list[InvoiceLine],
) -> None:
    """Enregistre le HT catalogue sur la ligne avant application remise globale (retrait si snapshot incomplet)."""
    for inv_ln in eligible:
        meta: dict[str, Any] = (
            dict(inv_ln.line_meta) if isinstance(inv_ln.line_meta, dict) else {}
        )
        meta[_META_ORIGINAL_LINE_TOTAL] = format(inv_ln.line_total or Decimal("0"), "f")
        inv_ln.line_meta = meta


def apply_draft_global_discount(
    company_id: int,
    invoice_id: int,
    *,
    global_discount_percent: float,
    global_discount_note: str | None = None,
    ride_line_ids: list[int] | None = None,
    expected_updated_at: str | None = None,
) -> EditDraftResult:
    """Applique une remise % sur le sous-total HT (transports concernés + prestations CUSTOM positives)."""
    inv, err_code, err_msg = _resolve_draft_invoice(
        company_id, invoice_id, expected_updated_at=expected_updated_at
    )
    if not inv:
        return EditDraftResult(
            False,
            error=_error_payload_from_resolve(err_msg),
            status_code=err_code or 400,
        )
    logger.info(
        "[draft_edit] apply_global_discount invoice_id=%s percent=%s company_id=%s",
        invoice_id,
        global_discount_percent,
        company_id,
    )

    if not (0 < global_discount_percent <= _MAX_DISCOUNT_PERCENT):
        return EditDraftResult(
            False,
            error={"error": "Remise: pourcentage invalide (0-100]"},
            status_code=400,
        )

    _undo_auto_discount_lines_and_restore_catalog_for_global_discount(inv)

    meta_clear: dict[str, Any] = dict(inv.meta) if isinstance(inv.meta, dict) else {}
    meta_clear.pop("global_discount", None)
    meta_clear.pop("per_line_discounts", None)
    inv.meta = meta_clear if meta_clear else None

    all_ride_lines = [ln for ln in inv.lines if ln.type == InvoiceLineType.RIDE]
    all_ride_ids = {ln.id for ln in all_ride_lines if ln.id is not None}

    if ride_line_ids is not None:
        if len(ride_line_ids) == 0:
            return EditDraftResult(
                False,
                error={
                    "error": "Indiquez au moins une ligne transport pour la remise %"
                },
                status_code=400,
            )
        try:
            selected_ids = {int(x) for x in ride_line_ids}
        except (TypeError, ValueError):
            return EditDraftResult(
                False, error={"error": "ride_line_ids invalide"}, status_code=400
            )
        if not selected_ids <= all_ride_ids:
            return EditDraftResult(
                False,
                error={"error": "Lignes transport invalides pour la remise"},
                status_code=400,
            )
        ride_lines = [ln for ln in all_ride_lines if ln.id in selected_ids]
        if not ride_lines:
            return EditDraftResult(
                False,
                error={"error": "Aucune ligne transport valide pour la remise"},
                status_code=400,
            )
    else:
        ride_lines = all_ride_lines

    custom_pos = _eligible_positive_custom_lines_for_global_discount(inv)
    eligible = sorted(ride_lines + custom_pos, key=lambda x: (x.id or 0))

    gross: Decimal = sum(
        (x.line_total or Decimal("0") for x in eligible),
        start=Decimal("0"),
    )
    if gross <= 0:
        return EditDraftResult(
            False,
            error={
                "error": "Aucun montant HT à remiser (transports et lignes positives)."
            },
            status_code=400,
        )

    gd = Decimal(str(global_discount_percent))
    discount_ht: Decimal = min(
        round_to_5_cents(gross * gd / Decimal("100")),
        gross,
    )
    net_ht = round_to_5_cents(gross - discount_ht)

    note = (global_discount_note or "").strip()
    # Catalogue avant remise (méta ligne) + snapshot JSON — même instantané que la capture.
    _stash_original_line_totals_for_global_discount(eligible)
    line_snapshots = [_line_ht_snapshot_dict(inv_ln) for inv_ln in eligible]

    running = Decimal("0.00")
    n = len(eligible)
    settings_repo = CompanyBillingSettingsRepository()
    bset = settings_repo.find_or_create(company_id)
    vat_rate = (
        Decimal(str(bset.vat_rate)) if bset.vat_rate is not None else Decimal("0.00")
    )
    vat_applicable = bool(bset.vat_applicable) and vat_rate > 0
    calc = InvoiceCalculator()

    for idx, lm in enumerate(eligible):
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

    db.session.flush()
    _recompute_totals_from_lines(inv)
    sum_eligible = sum((x.line_total or Decimal("0")) for x in eligible)
    if abs(sum_eligible - net_ht) > Decimal("0.02"):
        logger.warning(
            "[draft_edit] global_discount invariant HT eligible=%s net_ht=%s invoice_id=%s",
            sum_eligible,
            net_ht,
            invoice_id,
        )
    gd_meta: dict[str, Any] = {
        "percent": float(gd),
        "amount_ht": float(discount_ht),
        "subtotal_before_ht": float(gross),
        "line_snapshots": line_snapshots,
        "snapshot_version": 2,
    }
    if note:
        gd_meta["note"] = note
    if len(ride_lines) < len(all_ride_lines):
        gd_meta["ride_line_ids"] = [rl.id for rl in ride_lines if rl.id is not None]
    meta: dict[str, Any] = dict(inv.meta) if isinstance(inv.meta, dict) else {}
    meta.pop("per_line_discounts", None)
    meta["global_discount"] = gd_meta
    inv.meta = meta
    _mark_pdf_stale(inv)
    db.session.commit()
    return EditDraftResult(True, invoice=inv)


def apply_draft_per_line_discounts(
    company_id: int,
    invoice_id: int,
    *,
    line_discounts: list[dict[str, Any]],
    expected_updated_at: str | None = None,
) -> EditDraftResult:
    """Remises % par ligne : réduit les transports et prestations CUSTOM positives ciblés."""
    inv, err_code, err_msg = _resolve_draft_invoice(
        company_id, invoice_id, expected_updated_at=expected_updated_at
    )
    if not inv:
        return EditDraftResult(
            False,
            error=_error_payload_from_resolve(err_msg),
            status_code=err_code or 400,
        )
    logger.info(
        "[draft_edit] apply_per_line_discounts invoice_id=%s company_id=%s n=%s",
        invoice_id,
        company_id,
        len(line_discounts),
    )

    wanted: dict[int, Decimal] = {}
    for raw in line_discounts:
        raw_lid = raw.get("line_id")
        if raw_lid is None:
            continue
        try:
            lid = int(raw_lid)
        except (TypeError, ValueError):
            continue
        raw_pct = raw.get("percent")
        if raw_pct is None:
            continue
        try:
            pf = float(raw_pct)
        except (TypeError, ValueError):
            continue
        if not (0 < pf <= _MAX_DISCOUNT_PERCENT):
            continue
        wanted[lid] = Decimal(str(pf))

    if not wanted:
        return remove_draft_global_discount(
            company_id, invoice_id, expected_updated_at=expected_updated_at
        )

    # Toujours lever une remise % précédente (globale ou par ligne) avant réapplication — évite double remise silencieuse.
    _undo_auto_discount_lines_and_restore_catalog_for_global_discount(inv)
    err_custom_restore = _restore_positive_custom_lines_from_catalog_meta(
        inv, company_id
    )
    if err_custom_restore is not None:
        return err_custom_restore
    meta_pre: dict[str, Any] = dict(inv.meta) if isinstance(inv.meta, dict) else {}
    meta_pre.pop("global_discount", None)
    meta_pre.pop("per_line_discounts", None)
    inv.meta = meta_pre if meta_pre else None

    eligible_lines = _eligible_lines_for_per_line_discount(inv)
    eligible_line_ids = {el.id for el in eligible_lines}
    if not set(wanted.keys()) <= eligible_line_ids:
        return EditDraftResult(
            False,
            error={"error": "Lignes invalides pour la remise par ligne"},
            status_code=400,
        )

    settings_repo = CompanyBillingSettingsRepository()
    bset = settings_repo.find_or_create(company_id)
    vat_rate = (
        Decimal(str(bset.vat_rate)) if bset.vat_rate is not None else Decimal("0.00")
    )
    vat_applicable = bool(bset.vat_applicable) and vat_rate > 0
    calc = InvoiceCalculator()

    lines_meta: list[dict[str, Any]] = []

    for lm in eligible_lines:
        catalog = lm.line_total or Decimal("0")
        pct = wanted.get(lm.id)
        if pct is None:
            continue
        meta_pre_line: dict[str, Any] = (
            dict(lm.line_meta) if isinstance(lm.line_meta, dict) else {}
        )
        meta_pre_line[_META_ORIGINAL_LINE_TOTAL] = format(catalog, "f")
        meta_pre_line["per_line_discount_percent"] = float(pct)
        lm.line_meta = meta_pre_line
        disc = min(
            round_to_5_cents(catalog * pct / Decimal("100")),
            catalog,
        )
        new_ht = round_to_5_cents(catalog - disc)
        lm.line_total = new_ht
        lm.unit_price = new_ht / (lm.qty or Decimal("1"))
        if vat_applicable and vat_rate > 0:
            lm.vat_rate = vat_rate
            va, tw = calc.calculate_vat(new_ht, vat_rate)
            lm.vat_amount = va
            lm.total_with_vat = tw
        else:
            lm.vat_rate = None
            lm.vat_amount = Decimal("0.00")
            lm.total_with_vat = new_ht

        lines_meta.append({"line_id": lm.id, "percent": float(pct)})

    db.session.flush()
    _recompute_totals_from_lines(inv)
    meta_out: dict[str, Any] = dict(inv.meta) if isinstance(inv.meta, dict) else {}
    meta_out.pop("global_discount", None)
    pl_wrap: dict[str, Any] = {"lines": lines_meta}
    meta_out["per_line_discounts"] = pl_wrap
    inv.meta = cast(Any, meta_out)
    _mark_pdf_stale(inv)
    db.session.commit()
    return EditDraftResult(True, invoice=inv)


def _restore_positive_custom_lines_from_catalog_meta(
    inv: Invoice, company_id: int
) -> EditDraftResult | None:
    """Restaure les prestations CUSTOM HT> depuis `original_line_total` si la méta existe encore.

    Utilisé après retrait remise globale : si la ligne était dans `snap_ids`, on sautait auparavant
    cette étape en supposant que le snapshot JSON suffisait — or le snapshot peut ne pas réappliquer
    correctement toutes les lignes CUSTOM ; tant que le catalogue est conservé en méta, on l'applique.
    Si `original_line_total` a déjà été retiré par `_apply_line_ht_snapshot`, on ignore la ligne.
    """
    settings_repo = CompanyBillingSettingsRepository()
    bset = settings_repo.find_or_create(company_id)
    cr_vat = (
        Decimal(str(bset.vat_rate)) if bset.vat_rate is not None else Decimal("0.00")
    )
    vat_ok = bool(bset.vat_applicable) and cr_vat > 0
    calc = InvoiceCalculator()
    for lm in inv.lines:
        if lm.type != InvoiceLineType.CUSTOM:
            continue
        lt = lm.line_total or Decimal("0")
        if lt <= 0:
            continue
        meta = lm.line_meta if isinstance(lm.line_meta, dict) else {}
        raw_ot = meta.get(_META_ORIGINAL_LINE_TOTAL)
        if raw_ot is None:
            continue
        try:
            restored = round_to_5_cents(Decimal(str(raw_ot)))
        except Exception:
            return EditDraftResult(
                False,
                error={
                    "error": (
                        "Retrait remise : montant catalogue invalide pour une ligne prestation."
                    )
                },
                status_code=409,
            )
        lm.line_total = restored
        lm.unit_price = restored / (lm.qty or Decimal("1"))
        if vat_ok:
            lm.vat_rate = cr_vat
            va, tw = calc.calculate_vat(restored, cr_vat)
            lm.vat_amount = va
            lm.total_with_vat = tw
        else:
            lm.vat_rate = None
            lm.vat_amount = Decimal("0.00")
            lm.total_with_vat = restored
        meta2 = dict(lm.line_meta) if isinstance(lm.line_meta, dict) else {}
        meta2.pop(_META_ORIGINAL_LINE_TOTAL, None)
        meta2.pop("per_line_discount_percent", None)
        lm.line_meta = meta2 if meta2 else None
    return None


def _restore_single_ride_line_from_booking(lm: InvoiceLine, company_id: int) -> None:
    calc = InvoiceCalculator()
    settings_repo = CompanyBillingSettingsRepository()
    bset = settings_repo.find_or_create(company_id)
    vat_rate = bset.vat_rate or Decimal("0.00")
    vat_applicable = bool(bset.vat_applicable) and (vat_rate > 0)
    reservation_id = getattr(lm, "reservation_id", None)
    bk = Booking.query.get(reservation_id) if reservation_id else None
    if not bk:
        return
    amt = getattr(bk, "amount", None)
    est = getattr(bk, "estimated_amount", None)
    raw = amt if amt is not None else (est if est is not None else 0)
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


def remove_draft_global_discount(
    company_id: int,
    invoice_id: int,
    *,
    expected_updated_at: str | None = None,
) -> EditDraftResult:
    """Supprime les remises % (méta globale et par ligne) : restaure snapshot complet ou Booking / méta catalogue."""
    inv, err_code, err_msg = _resolve_draft_invoice(
        company_id, invoice_id, expected_updated_at=expected_updated_at
    )
    if not inv:
        return EditDraftResult(
            False,
            error=_error_payload_from_resolve(err_msg),
            status_code=err_code or 400,
        )
    logger.info(
        "[draft_edit] remove_global_discount invoice_id=%s company_id=%s",
        invoice_id,
        company_id,
    )

    prev_snaps = _line_snapshots_from_global_discount_meta(inv)
    _delete_negative_custom_discount_lines_except_manual(inv)
    snap_ids: set[int] = set()
    if prev_snaps:
        for s in prev_snaps:
            try:
                snap_ids.add(int(s["id"]))
            except (KeyError, TypeError, ValueError):
                continue
        _restore_invoice_lines_from_global_discount_snapshots(inv, prev_snaps)
        # Filet : lignes transport encore avec `original_line_total` (snapshot sans entrée fiable, ids,
        # ou paire A/R partielle). Sinon le PDF garde « catalogue → net » et le HT reste remisé.
        err_rides_snap = _restore_positive_ride_lines_from_catalog_meta(inv, company_id)
        if err_rides_snap is not None:
            return err_rides_snap
        # Legacy : snapshot incomplet — RIDE absents → réservation
        for lm in inv.lines:
            if (
                lm.type != InvoiceLineType.RIDE
                or not lm.reservation_id
                or lm.id is None
            ):
                continue
            if lm.id in snap_ids:
                continue
            lm_meta = lm.line_meta if isinstance(lm.line_meta, dict) else {}
            if lm_meta.get(_META_ORIGINAL_LINE_TOTAL) is not None:
                continue
            _restore_single_ride_line_from_booking(lm, company_id)
        # CUSTOM HT> : Catalogue conservé en `original_line_total` (toutes lignes éligibles, y compris
        # celles listées dans le snapshot JSON — si le snapshot n'a pas tout rétabli, la méta suffit).
        err_custom = _restore_positive_custom_lines_from_catalog_meta(inv, company_id)
        if err_custom is not None:
            return err_custom
    else:
        meta_now = inv.meta if isinstance(inv.meta, dict) else {}
        # Sans remise % en méta, ne pas réécrire les lignes transport depuis Booking : le montant
        # facturé peut différer (ex. A/R, ajustements) — « Retirer les remises » ne doit pas casser le HT.
        if meta_now.get("global_discount") or meta_now.get("per_line_discounts"):
            ride_ids_with_catalog_meta = {
                lm.id
                for lm in inv.lines
                if lm.type == InvoiceLineType.RIDE
                and lm.id is not None
                and isinstance(lm.line_meta, dict)
                and lm.line_meta.get(_META_ORIGINAL_LINE_TOTAL) is not None
            }
            err_rides = _restore_positive_ride_lines_from_catalog_meta(inv, company_id)
            if err_rides is not None:
                return err_rides
            for lm in inv.lines:
                if (
                    lm.type != InvoiceLineType.RIDE
                    or not lm.reservation_id
                    or lm.id is None
                ):
                    continue
                if lm.id in ride_ids_with_catalog_meta:
                    continue
                _restore_single_ride_line_from_booking(lm, company_id)
        err_custom = _restore_positive_custom_lines_from_catalog_meta(inv, company_id)
        if err_custom is not None:
            return err_custom
    db.session.flush()
    db.session.expire(inv, ["lines"])
    _recompute_totals_from_lines(inv)
    meta = dict(inv.meta) if isinstance(inv.meta, dict) else {}
    meta.pop("global_discount", None)
    meta.pop("per_line_discounts", None)
    inv.meta = meta if meta else None
    _mark_pdf_stale(inv)
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
    expected_updated_at: str | None = None,
    service_date_iso: str | None = None,
) -> EditDraftResult:
    """Ajoute une ligne HT personnalisée (ex. accompagnement, forfait) — type CUSTOM, sans réservation.

    Montant HT strictement positif : prestation. Montant strictement négatif : remise libre (libellé au choix),
    conservée si la remise globale % est retirée ou réappliquée (méta `manual_discount`).
    """
    inv, err_code, err_msg = _resolve_draft_invoice(
        company_id, invoice_id, expected_updated_at=expected_updated_at
    )
    if not inv:
        return EditDraftResult(
            False,
            error=_error_payload_from_resolve(err_msg),
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
    if lt == 0:
        return EditDraftResult(
            False,
            error={"error": "Le montant HT ne peut pas être zéro"},
            status_code=400,
        )

    is_manual_discount = lt < 0
    line_meta: dict[str, Any] | None
    if is_manual_discount:
        q = Decimal("1")
        unit = lt
        line_meta = {_META_MANUAL_DISCOUNT: True}
    else:
        try:
            q = Decimal(str(qty))
        except Exception:
            q = Decimal("1")
        if q <= 0:
            q = Decimal("1")
        unit = round_to_5_cents(lt / q) if q else lt

        line_meta = None
        if custom_mode in ("time", "quantity"):
            entry: dict[str, Any] = {"mode": str(custom_mode)}
            if custom_mode == "time" and time_unit in ("min", "h", "d", "mois"):
                entry["time_unit"] = str(time_unit)
            line_meta = {"custom_prestation": entry}

    norm_service: str | None = None
    if service_date_iso is not None:
        sd_s = str(service_date_iso).strip()
        if sd_s:
            norm_service = _normalize_optional_service_date_iso(sd_s)
            if norm_service is None:
                return EditDraftResult(
                    False,
                    error={
                        "error": "service_date_iso invalide (format YYYY-MM-DD attendu).",
                    },
                    status_code=400,
                )
    if norm_service:
        if line_meta is None:
            line_meta = {"service_date_iso": norm_service}
        else:
            line_meta = {**line_meta, "service_date_iso": norm_service}

    logger.info(
        "[draft_edit] add_custom_line invoice_id=%s company_id=%s",
        invoice_id,
        company_id,
    )
    settings_repo = CompanyBillingSettingsRepository()
    bset = settings_repo.find_or_create(company_id)
    vat_rate = (
        Decimal(str(bset.vat_rate)) if bset.vat_rate is not None else Decimal("0.00")
    )
    vat_applicable = bool(bset.vat_applicable) and vat_rate > 0
    calc = InvoiceCalculator()

    line = InvoiceLine()
    line.invoice_id = inv.id
    line.type = InvoiceLineType.CUSTOM
    line.description = desc
    line.qty = q
    line.unit_price = unit
    line.line_total = lt
    line.vat_rate = None
    line.vat_amount = Decimal("0.00")
    line.total_with_vat = lt
    line.adjustment_note = None
    line.reservation_id = None
    line.line_meta = line_meta
    if vat_applicable and vat_rate > 0 and not is_manual_discount:
        line.vat_rate = vat_rate
        va, tw = calc.calculate_vat(lt, vat_rate)
        line.vat_amount = va
        line.total_with_vat = tw
    db.session.add(line)
    db.session.flush()
    # Sans expire, `inv.lines` peut rester le snapshot pré-flush (sans la nouvelle ligne) :
    # sous-totaux/TTC faux en DB alors que le PDF liste bien la remise (rechargement lignes).
    db.session.expire(inv, ["lines"])
    _recompute_totals_from_lines(inv)
    _mark_pdf_stale(inv)
    db.session.commit()
    return EditDraftResult(True, invoice=inv)
